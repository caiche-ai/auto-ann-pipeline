import json
import unittest
from pathlib import Path

from annotation_service.app import create_app
from annotation_service.config import Settings


def operation_ids(document: dict) -> set[str]:
    return {
        operation["operationId"]
        for path in document["paths"].values()
        for method, operation in path.items()
        if method in {"get", "post", "put", "patch", "delete"}
    }


class StaticOpenAPIContractTest(unittest.TestCase):
    def test_static_contract_matches_runtime_operations(self):
        static_path = (
            Path(__file__).resolve().parents[2]
            / "docs_caich"
            / "annotation_openapi.yaml"
        )
        static = json.loads(static_path.read_text(encoding="utf-8"))
        runtime = create_app(
            Settings(
                docs_enabled=True,
                storage_enabled=False,
            )
        ).openapi()

        self.assertEqual(static["openapi"], "3.0.3")
        self.assertEqual(operation_ids(static), operation_ids(runtime))
        self.assertEqual(
            operation_ids(static),
            {
                "getHealth",
                "getReadiness",
                "createAsset",
                "getAsset",
                "getAssetContent",
                "createAnnotationJob",
                "getAnnotationJob",
                "cancelAnnotationJob",
                "listAnnotationJobDetections",
                "getAnnotationJobBoundingBoxImage",
                "buildDetectionReviewTasks",
                "listAnnotationTasks",
                "getAnnotationTask",
                "listAnnotationTaskVersions",
                "listAnnotationTaskReviews",
                "saveAnnotationDraft",
                "submitAnnotationTask",
                "invalidateAnnotationTask",
                "reviewAnnotationTask",
                "createMaskCandidate",
                "createBatchMaskCandidates",
                "createPromptEnrichment",
                "createBatchPromptEnrichments",
                "createJointPromptEnrichment",
                "getAnnotationTaskGroup",
                "getTaskArtifact",
                "getAnnotationOperation",
                "cancelAnnotationOperation",
                "createAnnotationRelease",
                "getAnnotationRelease",
                "getAnnotationReleaseManifest",
                "getAnnotationReleaseArchive",
            },
        )
        static_request = static["components"]["schemas"][
            "CreateJobRequest"
        ]["properties"]
        runtime_request = runtime["components"]["schemas"][
            "CreateJobRequest"
        ]["properties"]
        for field in (
            "grounding_prompt_normalization_mode",
            "grounding_prompt_normalization_profile",
            "grounding_prompt_translation_failure_policy",
        ):
            self.assertEqual(
                static_request[field]["enum"],
                runtime_request[field]["enum"],
            )
            self.assertEqual(
                static_request[field]["default"],
                runtime_request[field]["default"],
            )
        for schema_name in ("GroundingPromptRoute", "Job"):
            static_schema = static["components"]["schemas"][schema_name]
            runtime_schema = runtime["components"]["schemas"][schema_name]
            self.assertEqual(
                set(static_schema.get("properties", {})),
                set(runtime_schema.get("properties", {})),
            )
            self.assertEqual(
                set(static_schema.get("required", [])),
                set(runtime_schema.get("required", [])),
            )
        for schema_name in (
            "BatchMaskCandidateItem",
            "BatchMaskCandidatesRequest",
            "BatchOperationItemResult",
            "BatchOperationsAccepted",
            "BatchPromptEnrichmentItem",
            "BatchPromptEnrichmentsRequest",
            "BuildReviewTasksResponse",
            "DetectionOverlapWarning",
            "JointPromptEnrichmentRequest",
            "ReviewTaskBuildItem",
            "TaskGroup",
            "TaskGroupMember",
            "TaskGroupOperationAccepted",
            "AnnotationOperation",
        ):
            static_schema = static["components"]["schemas"][schema_name]
            runtime_schema = runtime["components"]["schemas"][schema_name]
            self.assertEqual(
                set(static_schema.get("properties", {})),
                set(runtime_schema.get("properties", {})),
            )
            self.assertEqual(
                set(static_schema.get("required", [])),
                set(runtime_schema.get("required", [])),
            )


if __name__ == "__main__":
    unittest.main()
