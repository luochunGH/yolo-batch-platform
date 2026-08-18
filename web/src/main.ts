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
}

const statusText: Record<string, string> = {
  queued: '排队中', running: '处理中', cancelling: '取消中', completed: '已完成', failed: '失败', cancelled: '已取消',
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
    const upload = async () => {
      if (!archive.value) { message.value = mode.value === 'inference' ? '请先选择图片 ZIP' : '请先选择带标注数据集 ZIP'; return }
      if (!isTrain.value && !trainedModelId.value) { message.value = '请先选择已训练模型'; return }
      uploading.value = true
      try {
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
        archive.value = null; name.value = ''
        if (fileInput.value) fileInput.value.value = ''
        await refresh()
      } catch (error: any) { message.value = error.message } finally { uploading.value = false }
    }
    const cancel = async (id: string) => { await request(`/jobs/${id}/cancel`, { method: 'POST' }); await refresh() }
    const remove = async (id: string) => { if (confirm('确定删除该任务及其结果文件吗？')) { await request(`/jobs/${id}`, { method: 'DELETE' }); await refresh() } }
    const download = async (job: Job) => {
      const response = await request(`/jobs/${job.id}/download`)
      const blob = await response.blob(); const url = URL.createObjectURL(blob)
      const suffix = job.task_type === 'train' ? 'best.pt' : job.task_type === 'evaluate' ? 'evaluation.json' : 'inference.zip'
      const link = document.createElement('a'); link.href = url; link.download = `${job.name}-${suffix}`; link.click(); URL.revokeObjectURL(url)
    }
    const modelLabel = (job: Job) => job.task_type === 'train' ? job.model : trainedModels.value.find((item: any) => item.id === job.model_id)?.name || '已训练模型'
    onMounted(() => { refresh(); window.setInterval(refresh, 2000) })
    return { apiKey, jobs, sortedJobs, dashboard, uploading, message, archive, fileInput, connectionText, connectionState, onApiKeyInput, name, mode, modeItems, isTrain, model, modelOptions, availableModels, trainedModels, trainedModelId, imgszMode, customWidth, customHeight, confidence, epochs, trainBatch, active, statusText, taskTypeText, modelLabel, saveKey, selectFile, upload, cancel, remove, download, refresh }
  },
  template: `
    <main class="shell">
      <p v-if="message" class="message">{{ message }}</p>
      <section class="workspace-grid">
        <article class="panel service-panel"><div class="panel-title"><h2>服务状态</h2><span class="worker-light" :class="['training','evaluating','inferring'].includes(dashboard.worker.state) ? 'busy' : 'ready'"></span></div><div class="worker-connection"><span class="status-dot" :class="connectionState === 'connected' ? 'online' : ''"></span><span>{{ connectionText }}</span><input v-model="apiKey" @input="onApiKeyInput" type="password" placeholder="输入 API Key"><button class="secondary" :disabled="connectionState === 'connected' || connectionState === 'checking'" @click="saveKey">{{ connectionState === 'connected' ? '已连接' : (connectionState === 'checking' ? '连接中' : '连接服务') }}</button></div><div class="worker-metrics"><div><small>进行中任务</small><strong>{{ dashboard.counts.running || 0 }}</strong></div><div><small>排队任务</small><strong>{{ dashboard.counts.queued || 0 }}</strong></div><div><small>已完成模型</small><strong>{{ dashboard.counts.completed || 0 }}</strong></div><div><small>GPU 利用率</small><strong>{{ dashboard.worker.gpu_utilization == null ? '—' : dashboard.worker.gpu_utilization + '%' }}</strong></div><div><small>GPU 型号</small><strong>{{ dashboard.worker.gpu_name || '—' }}</strong></div></div><div class="progress"><i :style="{width: (active?.progress || 0) + '%'}"></i></div><p class="progress-label">{{ active ? active.name + ' · ' + (taskTypeText[active.task_type] || '任务') + ' · ' + (active.completed || 0) + ' / ' + (active.total || 0) : '等待新的任务' }}</p><dl><div><dt>显存使用</dt><dd>{{ dashboard.worker.gpu_memory_used == null ? '—' : dashboard.worker.gpu_memory_used + ' / ' + dashboard.worker.gpu_memory_total + ' MiB' }}</dd></div><div><dt>GPU 温度</dt><dd>{{ dashboard.worker.gpu_temperature == null ? '—' : dashboard.worker.gpu_temperature + ' °C' }}</dd></div><div><dt>GPU 功耗</dt><dd>{{ dashboard.worker.gpu_power == null ? '—' : dashboard.worker.gpu_power + ' W' }}</dd></div><div><dt>磁盘可用</dt><dd>{{ Math.round((dashboard.disk?.free || 0) / 1073741824) }} GiB</dd></div><div><dt>Worker 心跳</dt><dd>{{ dashboard.worker.updated_at ? new Date(dashboard.worker.updated_at).toLocaleTimeString('zh-CN') : '—' }}</dd></div></dl></article>
        <article class="panel upload"><div class="panel-title"><h2>{{ isTrain ? '训练配置' : (mode === 'evaluate' ? '评估配置' : '推理配置') }}</h2><span v-if="active" class="badge running">{{ statusText[active.status] }}</span></div><div class="mode-switch"><button v-for="item in modeItems" :key="item.key" :class="{active: mode === item.key}" @click="mode = item.key">{{ item.label }}</button></div><label class="drop"><input ref="fileInput" type="file" accept=".zip" @change="selectFile"><span class="upload-mark">↑</span><b>{{ archive ? archive.name : (mode === 'inference' ? '选择图片 ZIP 压缩包' : '选择带标注数据集 ZIP 压缩包') }}</b><small v-if="mode !== 'inference'">需包含 data.yaml（train / val / names）、images/、labels/</small><small v-if="mode !== 'inference'">图片与同名 TXT 标注文件一一对应</small><small v-else>只需包含图片，不需要 data.yaml 和标注文件</small></label><div class="form-grid"><label>任务名称<input v-model="name" :placeholder="mode === 'train' ? '例如：缺陷检测模型-第一版' : (mode === 'evaluate' ? '例如：第一版模型评估' : '例如：产线图片推理')"></label><label v-if="isTrain">预训练权重<select v-model="model"><option v-for="option in modelOptions" :key="option" :value="option" :disabled="!availableModels.includes(option)">{{ option }}{{ availableModels.includes(option) ? '' : '（未安装）' }}</option></select></label><label v-else>已训练模型<select v-model="trainedModelId"><option value="" disabled>选择已完成训练模型</option><option v-for="item in trainedModels" :key="item.id" :value="item.id">{{ item.name }}（best.pt）</option></select></label><label>输入尺寸<select v-model="imgszMode"><option value="320">320</option><option value="640">640（推荐）</option><option value="960">960</option><option value="1280">1280</option><option value="original">原图</option><option value="custom">自定义</option></select></label><label v-if="imgszMode === 'custom'" class="size-pair">自定义宽 × 高<div><input v-model.number="customWidth" type="number" min="320" max="4096" step="32" placeholder="宽"><span>×</span><input v-model.number="customHeight" type="number" min="320" max="4096" step="32" placeholder="高"></div></label><label v-if="!isTrain">置信度阈值<input v-model.number="confidence" type="number" min="0.01" max="0.99" step="0.01"></label><label v-if="isTrain">训练轮数<input v-model.number="epochs" type="number" min="1" max="1000" step="10"></label><label v-if="isTrain">Batch Size<input v-model.number="trainBatch" type="number" min="1" max="128" step="1"></label></div><button class="primary" :disabled="uploading || connectionState !== 'connected' || (isTrain ? !availableModels.includes(model) : !trainedModelId)" @click="upload">{{ uploading ? '正在上传...' : (isTrain ? '开始训练' : (mode === 'evaluate' ? '开始评估' : '开始推理')) }}</button></article>
        <section class="panel table"><div class="panel-title"><h2>任务记录</h2><button class="secondary" @click="refresh">刷新</button></div><div class="task-list"><table class="task-table"><thead><tr><th>任务名称</th><th>类型</th><th>模型</th><th>状态</th><th>进度</th><th>创建时间</th><th>操作</th></tr></thead><tbody><tr v-for="job in sortedJobs" :key="job.id"><td class="task-name" :title="job.name">{{ job.name }}</td><td>{{ taskTypeText[job.task_type] || job.task_type }}</td><td class="task-model" :title="modelLabel(job)">{{ modelLabel(job) }}</td><td><span class="badge" :class="job.status">{{ statusText[job.status] || job.status }}</span></td><td class="task-progress"><div class="mini-progress"><i :style="{width: job.progress + '%'}"></i></div><small>{{ job.completed }}/{{ job.total }} · {{ job.progress }}%</small></td><td>{{ new Date(job.created_at).toLocaleString('zh-CN') }}</td><td class="actions"><button v-if="['queued','running','cancelling'].includes(job.status)" @click="cancel(job.id)">取消</button><button v-if="job.artifact_path || job.result_path" @click="download(job)">{{ job.task_type === 'train' ? '下载模型' : (job.task_type === 'evaluate' ? '下载报告' : '下载结果') }}</button><button v-if="!['queued','running','cancelling'].includes(job.status)" class="danger" @click="remove(job.id)">删除</button></td></tr></tbody></table><p v-if="!sortedJobs.length" class="empty">暂无任务记录</p></div></section>
      </section>
    </main>`
}

createApp(App).mount('#app')
