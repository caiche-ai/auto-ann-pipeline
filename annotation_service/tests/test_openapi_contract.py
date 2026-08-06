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
                "createPromptEnrichment",
                "getTaskArtifact",
                "getAnnotationOperation",
                "cancelAnnotationOperation",
                "createAnnotationRelease",
                "getAnnotationRelease",
                "getAnnotationReleaseManifest",
                "getAnnotationReleaseArchive",
            },
        )


if __name__ == "__main__":
    unittest.main()
