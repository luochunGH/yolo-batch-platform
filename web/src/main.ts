import { computed, createApp, onMounted, ref } from 'vue/dist/vue.esm-bundler.js'
import './style.css'

type Job = {
  id: string
  name: string
  task_type: string
  status: string
  model: string
  model_id?: string
  total: number
  completed: number
  failed: number
  progress: number
  created_at: string
  error?: string
  result_path?: string
  artifact_path?: string
  uploaded_path?: string
  imgsz: string | number
  confidence: number
  epochs: number
  train_batch: number
  started_at?: string
  finished_at?: string
}

const statusText: Record<string, string> = {
  queued: '排队中', running: '运行中', cancelling: '取消中', completed: '已完成', failed: '失败', cancelled: '已取消',
}
const taskTypeText: Record<string, string> = { train: '训练', evaluate: '评估', inference: '推理' }
const modeItems = [{ key: 'train', label: '训练' }, { key: 'evaluate', label: '评估' }, { key: 'inference', label: '推理' }]

const App = {
  setup() {
    const apiKey = ref(localStorage.getItem('yolo-api-key') || '')
    const jobs = ref<Job[]>([])
    const dashboard = ref<any>({ counts: {}, worker: {}, trained_models: [] })
    const uploading = ref(false)
    const message = ref('')
    const archive = ref<File | null>(null)
    const sourceJobId = ref('')
    const detailJob = ref<Job | null>(null)
    const fileInput = ref<HTMLInputElement | null>(null)
    const connectionState = ref<'idle' | 'checking' | 'connected' | 'error'>(apiKey.value ? 'checking' : 'idle')
    const mode = ref('train')
    const name = ref('')
    const model = ref('yolo11n.pt')
    const trainedModelId = ref('')
    const imgszMode = ref('640')
    const customWidth = ref(1280)
    const customHeight = ref(720)
    const confidence = ref(0.25)
    const imgszValue = computed(() => imgszMode.value === 'custom' ? `${customWidth.value}x${customHeight.value}` : imgszMode.value)
    const epochs = ref(50)
    const trainBatch = ref(16)
    const active = computed(() => jobs.value.find(job => ['running', 'queued', 'cancelling'].includes(job.status)))
    const displayStatus = (job: Job) => {
      if (job.status === 'queued') return '等待 GPU Worker'
      if (job.status === 'running' && dashboard.value.worker?.job_id === job.id && dashboard.value.worker?.phase) return dashboard.value.worker.phase
      if (job.status === 'running') return 'Worker 执行中'
      if (job.status === 'cancelling') return '正在停止任务'
      if (job.status === 'completed' && job.task_type === 'inference' && job.failed) return `已完成，跳过 ${job.failed} 张`
      return statusText[job.status] || job.status
    }
    const durationText = (job: Job) => {
      if (!job.started_at) return '—'
      const end = job.finished_at ? Date.parse(job.finished_at) : Date.now()
      const seconds = Math.max(0, Math.floor((end - Date.parse(job.started_at)) / 1000))
      const minutes = Math.floor(seconds / 60)
      return minutes ? `${minutes} 分 ${seconds % 60} 秒` : `${seconds} 秒`
    }
    const sortedJobs = computed(() => [...jobs.value].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at)))
    const modelOptions = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']
    const availableModels = computed(() => dashboard.value.models || [])
    const trainedModels = computed(() => dashboard.value.trained_models || [])
    const isTrain = computed(() => mode.value === 'train')

    const headers = () => ({ 'X-API-Key': apiKey.value })
    const request = async (path: string, options: RequestInit = {}) => {
      const response = await fetch(`/api/v1${path}`, { ...options, headers: { ...headers(), ...(options.headers || {}) } })
      if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || '请求失败，请检查服务状态')
      return response
    }
    const refresh = async () => {
      if (!apiKey.value) { connectionState.value = 'idle'; return }
      try {
        const [jobResponse, dashboardResponse] = await Promise.all([request('/jobs'), request('/dashboard')])
        jobs.value = await jobResponse.json()
        dashboard.value = await dashboardResponse.json()
        connectionState.value = 'connected'
      } catch (error: any) { connectionState.value = 'error'; message.value = error.message }
    }
    const connectionText = computed(() => ({ idle: '待连接', checking: '连接中', connected: '服务已连接', error: '连接失败' })[connectionState.value])
    const onApiKeyInput = () => { connectionState.value = 'idle' }
    const saveKey = async () => {
      if (!apiKey.value) { connectionState.value = 'idle'; message.value = '请输入 API Key'; return }
      connectionState.value = 'checking'
      try {
        const dashboardResponse = await request('/dashboard')
        dashboard.value = await dashboardResponse.json()
        localStorage.setItem('yolo-api-key', apiKey.value)
        connectionState.value = 'connected'
        message.value = '服务连接成功'
        await refresh()
      } catch (error: any) { connectionState.value = 'error'; message.value = error.message }
    }
    const selectFile = (event: Event) => { archive.value = (event.target as HTMLInputElement).files?.[0] || null }
    const showDetails = (job: Job) => { detailJob.value = job }
    const retry = async (job: Job, edit = false) => {
      if (edit) {
        mode.value = 'train'; sourceJobId.value = job.id; archive.value = null; name.value = `${job.name}-重新训练`
        model.value = job.model; imgszMode.value = String(job.imgsz); epochs.value = job.epochs || 50; trainBatch.value = job.train_batch || 16
        message.value = '已载入原任务参数，修改后点击“开始训练”即可复用原 ZIP'
        return
      }
      if (!confirm(`确定复制“${job.name}”并立即重新训练吗？`)) return
      const body = new FormData(); body.append('name', `${job.name}-重新训练`)
      await request(`/jobs/${job.id}/retry`, { method: 'POST', body }); message.value = '已复制 ZIP，重新训练任务已进入队列'; await refresh()
    }
    const upload = async () => {
      if (!sourceJobId.value && !archive.value) { message.value = mode.value === 'inference' ? '请先选择图片 ZIP' : '请先选择带标注数据集 ZIP'; return }
      if (!isTrain.value && !trainedModelId.value) { message.value = '请先选择已训练模型'; return }
      uploading.value = true
      try {
        if (sourceJobId.value) {
          const body = new FormData(); body.append('name', name.value); body.append('model', model.value); body.append('imgsz', imgszValue.value); body.append('epochs', String(epochs.value)); body.append('train_batch', String(trainBatch.value))
          await request(`/jobs/${sourceJobId.value}/retry`, { method: 'POST', body }); message.value = '已复制 ZIP，重新训练任务已进入队列'; sourceJobId.value = ''; name.value = ''; await refresh(); return
        }
        const body = new FormData()
        body.append('archive', archive.value)
        body.append('name', name.value)
        body.append('task_type', mode.value)
        body.append('model', isTrain.value ? model.value : 'yolo11n.pt')
        if (!isTrain.value) body.append('model_id', trainedModelId.value)
        body.append('imgsz', imgszValue.value)
        body.append('confidence', String(confidence.value))
        body.append('epochs', String(epochs.value))
        body.append('train_batch', String(trainBatch.value))
        await request('/jobs', { method: 'POST', body })
        message.value = `${taskTypeText[mode.value]}任务已创建并进入 GPU 队列`
        archive.value = null; name.value = ''; sourceJobId.value = ''
        if (fileInput.value) fileInput.value.value = ''
        await refresh()
      } catch (error: any) { message.value = error.message } finally { uploading.value = false }
    }
    const cancel = async (id: string) => { await request(`/jobs/${id}/cancel`, { method: 'POST' }); await refresh() }
    const remove = async (id: string) => { if (confirm('确定删除该任务及其结果文件吗？')) { await request(`/jobs/${id}`, { method: 'DELETE' }); await refresh() } }
    const download = async (job: Job) => {
      const response = await request(`/jobs/${job.id}/download`)
      const blob = await response.blob(); const url = URL.createObjectURL(blob)
      const suffix = job.task_type === 'train' ? 'best.pt' : job.task_type === 'evaluate' ? 'evaluation.json' : 'inference.csv'
      const link = document.createElement('a'); link.href = url; link.download = `${job.name}-${suffix}`; link.click(); URL.revokeObjectURL(url)
    }
    const modelLabel = (job: Job) => job.task_type === 'train' ? job.model : trainedModels.value.find((item: any) => item.id === job.model_id)?.name || '已训练模型'
    onMounted(() => { refresh(); window.setInterval(refresh, 2000) })
    return { apiKey, jobs, sortedJobs, dashboard, uploading, message, archive, sourceJobId, detailJob, fileInput, connectionText, connectionState, onApiKeyInput, name, mode, modeItems, isTrain, model, modelOptions, availableModels, trainedModels, trainedModelId, imgszMode, customWidth, customHeight, confidence, epochs, trainBatch, active, statusText, taskTypeText, displayStatus, durationText, modelLabel, saveKey, selectFile, upload, cancel, remove, download, refresh, showDetails, retry }
  },
  template: `
    <main class="shell">
      <p v-if="message" class="message">{{ message }}</p>
      <section class="workspace-grid">
        <article class="panel service-panel"><div class="panel-title"><h2>服务状态</h2><span class="worker-light" :class="['training','evaluating','inferring'].includes(dashboard.worker.state) ? 'busy' : 'ready'"></span></div><div class="worker-connection"><span class="status-dot" :class="connectionState === 'connected' ? 'online' : ''"></span><span>{{ connectionText }}</span><input v-model="apiKey" @input="onApiKeyInput" type="password" placeholder="输入 API Key"><button class="secondary" :disabled="connectionState === 'connected' || connectionState === 'checking'" @click="saveKey">{{ connectionState === 'connected' ? '已连接' : (connectionState === 'checking' ? '连接中' : '连接服务') }}</button></div><div class="worker-metrics"><div><small>进行中任务</small><strong>{{ dashboard.counts.running || 0 }}</strong></div><div><small>排队任务</small><strong>{{ dashboard.counts.queued || 0 }}</strong></div><div><small>已完成模型</small><strong>{{ dashboard.counts.completed || 0 }}</strong></div><div><small>GPU 利用率</small><strong>{{ dashboard.worker.gpu_utilization == null ? '—' : dashboard.worker.gpu_utilization + '%' }}</strong></div><div><small>GPU 型号</small><strong>{{ dashboard.worker.gpu_name || '—' }}</strong></div></div><div class="progress"><i :style="{width: (active?.progress || 0) + '%'}"></i></div><p class="progress-label">{{ active ? active.name + ' · ' + (dashboard.worker.job_id === active.id && dashboard.worker.phase ? dashboard.worker.phase : displayStatus(active)) : '等待新的任务' }}</p><dl><div><dt>显存使用</dt><dd>{{ dashboard.worker.gpu_memory_used == null ? '—' : dashboard.worker.gpu_memory_used + ' / ' + dashboard.worker.gpu_memory_total + ' MiB' }}</dd></div><div><dt>GPU 温度</dt><dd>{{ dashboard.worker.gpu_temperature == null ? '—' : dashboard.worker.gpu_temperature + ' °C' }}</dd></div><div><dt>GPU 功耗</dt><dd>{{ dashboard.worker.gpu_power == null ? '—' : dashboard.worker.gpu_power + ' W' }}</dd></div><div><dt>磁盘可用</dt><dd>{{ Math.round((dashboard.disk?.free || 0) / 1073741824) }} GiB</dd></div><div><dt>Worker 心跳</dt><dd>{{ dashboard.worker.updated_at ? new Date(dashboard.worker.updated_at).toLocaleTimeString('zh-CN') : '—' }}</dd></div></dl></article>
        <article class="panel upload"><div class="panel-title"><h2>{{ isTrain ? '训练配置' : (mode === 'evaluate' ? '评估配置' : '推理配置') }}</h2><span v-if="active" class="badge running">{{ displayStatus(active) }}</span></div><div class="mode-switch"><button v-for="item in modeItems" :key="item.key" :class="{active: mode === item.key}" @click="mode = item.key">{{ item.label }}</button></div><label class="drop"><input v-if="!sourceJobId" ref="fileInput" type="file" accept=".zip" @change="selectFile"><span class="upload-mark">↑</span><b>{{ sourceJobId ? '将复用原任务 ZIP' : (archive ? archive.name : (mode === 'inference' ? '选择图片 ZIP 压缩包' : '选择带标注数据集 ZIP 压缩包')) }}</b><small v-if="sourceJobId">服务端会复制原任务 ZIP，原任务不受影响</small><small v-else-if="mode !== 'inference'">需包含 data.yaml（train / val / names）、images/、labels/</small><small v-else>只需包含图片，不需要 data.yaml 和标注文件</small></label><div class="form-grid"><label>任务名称<input v-model="name" :placeholder="mode === 'train' ? '例如：缺陷检测模型-第一版' : (mode === 'evaluate' ? '例如：第一版模型评估' : '例如：产线图片推理')"></label><label v-if="isTrain">预训练权重<select v-model="model"><option v-for="option in modelOptions" :key="option" :value="option" :disabled="!availableModels.includes(option)">{{ option }}{{ availableModels.includes(option) ? '' : '（未安装）' }}</option></select></label><label v-else>已训练模型<select v-model="trainedModelId"><option value="" disabled>选择已完成训练模型</option><option v-for="item in trainedModels" :key="item.id" :value="item.id">{{ item.name }}（best.pt）</option></select></label><label>输入尺寸<select v-model="imgszMode"><option value="320">320</option><option value="640">640（推荐）</option><option value="960">960</option><option value="1280">1280</option><option value="original">原图</option><option value="custom">自定义</option></select></label><label v-if="imgszMode === 'custom'" class="size-pair">自定义宽 × 高<div><input v-model.number="customWidth" type="number" min="320" max="4096" step="32" placeholder="宽"><span>×</span><input v-model.number="customHeight" type="number" min="320" max="4096" step="32" placeholder="高"></div></label><label v-if="!isTrain">置信度阈值<input v-model.number="confidence" type="number" min="0.01" max="0.99" step="0.01"></label><label v-if="isTrain">训练轮数<input v-model.number="epochs" type="number" min="1" max="1000" step="10"></label><label v-if="isTrain">Batch Size<input v-model.number="trainBatch" type="number" min="1" max="128" step="1"></label></div><button class="primary" :disabled="uploading || connectionState !== 'connected' || (isTrain ? !availableModels.includes(model) : !trainedModelId)" @click="upload">{{ uploading ? '正在处理...' : (sourceJobId ? '复制 ZIP 并重新训练' : (isTrain ? '开始训练' : (mode === 'evaluate' ? '开始评估' : '开始推理'))) }}</button></article>
        <section class="panel table"><div class="panel-title"><h2>任务记录</h2><button class="secondary" @click="refresh">刷新</button></div><div class="task-list"><table class="task-table"><thead><tr><th>任务名称</th><th>类型</th><th>模型</th><th>状态</th><th>创建时间</th><th>操作</th></tr></thead><tbody><tr v-for="job in sortedJobs" :key="job.id"><td class="task-name" :title="job.name">{{ job.name }}</td><td>{{ taskTypeText[job.task_type] || job.task_type }}</td><td class="task-model" :title="modelLabel(job)">{{ modelLabel(job) }}</td><td><span class="badge" :class="job.status">{{ displayStatus(job) }}</span></td><td>{{ new Date(job.created_at).toLocaleString('zh-CN') }}</td><td class="actions"><button @click="showDetails(job)">详情</button><button v-if="job.task_type === 'train' && !['queued','running','cancelling'].includes(job.status)" @click="retry(job, true)">修改参数</button><button v-if="job.artifact_path || job.result_path" @click="download(job)">{{ job.task_type === 'train' ? '下载模型' : (job.task_type === 'evaluate' ? '下载报告' : '下载结果') }}</button><button v-if="['queued','running','cancelling'].includes(job.status)" @click="cancel(job.id)">取消</button><button v-if="!['queued','running','cancelling'].includes(job.status)" class="danger" @click="remove(job.id)">删除</button></td></tr></tbody></table><p v-if="!sortedJobs.length" class="empty">暂无任务记录</p></div></section>
      </section>
    </main>
    <div v-if="detailJob" class="modal-backdrop" @click.self="detailJob = null"><article class="modal panel"><div class="panel-title"><h2>任务详情</h2><button class="icon-close" title="关闭" @click="detailJob = null">×</button></div><div class="detail-title"><strong>{{ detailJob.name }}</strong><span class="badge" :class="detailJob.status">{{ displayStatus(detailJob) }}</span></div><dl class="detail-grid"><div><dt>任务类型</dt><dd>{{ taskTypeText[detailJob.task_type] || detailJob.task_type }}</dd></div><div><dt>模型</dt><dd>{{ modelLabel(detailJob) }}</dd></div><div><dt>输入尺寸</dt><dd>{{ detailJob.imgsz }}</dd></div><div><dt>训练轮数</dt><dd>{{ detailJob.epochs || '—' }}</dd></div><div><dt>Batch Size</dt><dd>{{ detailJob.train_batch || '—' }}</dd></div><div v-if="detailJob.task_type === 'inference'"><dt>跳过图片</dt><dd>{{ detailJob.failed || 0 }}</dd></div><div><dt>创建时间</dt><dd>{{ new Date(detailJob.created_at).toLocaleString('zh-CN') }}</dd></div><div><dt>开始时间</dt><dd>{{ detailJob.started_at ? new Date(detailJob.started_at).toLocaleString('zh-CN') : '—' }}</dd></div><div><dt>结束时间</dt><dd>{{ detailJob.finished_at ? new Date(detailJob.finished_at).toLocaleString('zh-CN') : '—' }}</dd></div><div><dt>耗时</dt><dd>{{ durationText(detailJob) }}</dd></div></dl><div v-if="detailJob.error" class="error-box"><b>失败原因</b><p>{{ detailJob.error }}</p></div><div class="modal-actions"><button v-if="detailJob.task_type === 'train'" class="secondary" @click="retry(detailJob, true); detailJob = null">修改参数</button><button v-if="detailJob.artifact_path || detailJob.result_path" class="primary compact" @click="download(detailJob)">下载结果</button></div></article></div>
    <footer class="site-footer">
      <span>Docker YOLO Web Console · v1.0.0</span>
      <span class="footer-separator">·</span>
      <span>MIT License</span>
      <a class="github-link" href="https://github.com/luochunGH/yolo-batch-platform" target="_blank" rel="noreferrer" title="打开 GitHub 项目" aria-label="打开 GitHub 项目">
        <svg class="github-icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
        <span>GitHub</span>
      </a>
    </footer>`
}

createApp(App).mount('#app')
