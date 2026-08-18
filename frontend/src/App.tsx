import { Route, Routes } from 'react-router-dom'
import Layout from './Layout'
import DashboardPage from './pages/DashboardPage'
import IntakePage from './pages/IntakePage'
import JobsPage from './pages/JobsPage'
import TaskDetailPage from './pages/TaskDetailPage'
import TasksPage from './pages/TasksPage'
import ReleasesPage from './pages/ReleasesPage'
import SettingsPage from './pages/SettingsPage'

export default function App() {
  return <Routes><Route element={<Layout />}>
    <Route index element={<DashboardPage />} />
    <Route path="intake" element={<IntakePage />} />
    <Route path="jobs" element={<JobsPage />} />
    <Route path="tasks" element={<TasksPage />} />
    <Route path="tasks/:taskId" element={<TaskDetailPage />} />
    <Route path="releases" element={<ReleasesPage />} />
    <Route path="settings" element={<SettingsPage />} />
    <Route path="*" element={<DashboardPage />} />
  </Route></Routes>
}
