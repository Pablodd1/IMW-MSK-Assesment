// Voice Command Controller
// Hands-free control for medical assessment platform

interface VoiceCommandConfig {
  continuous: boolean
  interimResults: boolean
  language: string
  maxAlternatives: number
}

interface VoiceAction {
  command: string
  aliases: string[]
  action: () => void | Promise<void>
  feedback: string
}

class VoiceCommandController {
  private recognition: SpeechRecognition | null = null
  private isListening: boolean = false
  private commands: Map<string, VoiceAction> = new Map()
  private feedbackElement: HTMLElement | null = null
  private statusElement: HTMLElement | null = null
  private isSupported: boolean = false

  constructor() {
    this.initRecognition()
  }

  private initRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    
    if (!SpeechRecognition) {
      console.warn('Speech recognition not supported')
      this.isSupported = false
      return
    }

    this.recognition = new SpeechRecognition()
    this.recognition.continuous = false
    this.recognition.interimResults = true
    this.recognition.lang = 'en-US'
    this.recognition.maxAlternatives = 3

    this.recognition.onstart = () => {
      this.isListening = true
      this.updateStatus('listening')
    }

    this.recognition.onend = () => {
      this.isListening = false
      this.updateStatus('idle')
    }

    this.recognition.onerror = (event) => {
      console.error('Voice recognition error:', event.error)
      this.updateStatus('error')
    }

    this.recognition.onresult = (event) => {
      this.handleResult(event)
    }

    this.isSupported = true
    this.registerDefaultCommands()
  }

  private handleResult(event: SpeechRecognitionEvent) {
    const results = event.results
    const lastResult = results[results.length - 1]
    const transcript = lastResult[0].transcript.toLowerCase().trim()
    const confidence = lastResult[0].confidence

    this.showFeedback(`Heard: "${transcript}"`)

    if (lastResult.isFinal) {
      this.processCommand(transcript, confidence)
    }
  }

  private processCommand(transcript: string, confidence: number) {
    // Check exact matches first
    for (const [key, cmd] of this.commands.entries()) {
      if (transcript.includes(key) || cmd.aliases.some(a => transcript.includes(a))) {
        this.executeCommand(cmd)
        return
      }
    }

    // Try partial matching
    const words = transcript.split(' ')
    for (const [key, cmd] of this.commands.entries()) {
      for (const word of words) {
        if (key.includes(word) || word.includes(key)) {
          this.executeCommand(cmd)
          return
        }
      }
    }

    // No match - try AI interpretation
    this.interpretCommand(transcript)
  }

  private async interpretCommand(transcript: string) {
    // Simple keyword-based interpretation
    const keywords: Record<string, string> = {
      'start': 'start recording',
      'stop': 'stop recording',
      'record': 'start recording',
      'analyze': 'analyze movement',
      'flip': 'flip camera',
      'patient': 'show patients',
      'note': 'generate notes',
      'save': 'save assessment',
      'next': 'next step',
      'back': 'previous step',
      'help': 'show help'
    }

    for (const [keyword, command] of Object.entries(keywords)) {
      if (transcript.includes(keyword)) {
        const cmd = this.commands.get(command)
        if (cmd) {
          this.executeCommand(cmd)
          return
        }
      }
    }

    this.showFeedback(`Command not recognized: "${transcript}"`)
  }

  private executeCommand(cmd: VoiceAction) {
    this.showFeedback(cmd.feedback)
    try {
      const result = cmd.action()
      if (result instanceof Promise) {
        result.catch(err => console.error('Command error:', err))
      }
    } catch (err) {
      console.error('Command execution error:', err)
    }
  }

  private registerDefaultCommands() {
    this.registerCommand({
      command: 'start recording',
      aliases: ['begin recording', 'start record', 'begin record', 'record'],
      action: () => {
        if (typeof startRecording === 'function') startRecording()
      },
      feedback: '🎬 Recording started'
    })

    this.registerCommand({
      command: 'stop recording',
      aliases: ['stop record', 'end recording', 'end record', 'stop'],
      action: () => {
        if (typeof stopRecording === 'function') stopRecording()
      },
      feedback: '⏹️ Recording stopped'
    })

    this.registerCommand({
      command: 'analyze movement',
      aliases: ['analyze', 'analysis', 'run analysis', 'process'],
      action: () => {
        if (typeof analyzeMovement === 'function') analyzeMovement()
      },
      feedback: '🔍 Analyzing movement...'
    })

    this.registerCommand({
      command: 'flip camera',
      aliases: ['switch camera', 'change camera', 'flip'],
      action: () => {
        if (typeof flipCamera === 'function') flipCamera()
      },
      feedback: '🔄 Camera flipped'
    })

    this.registerCommand({
      command: 'show patients',
      aliases: ['open patients', 'view patients', 'patients'],
      action: () => {
        window.location.href = '/static/patients.html'
      },
      feedback: '📋 Opening patients...'
    })

    this.registerCommand({
      command: 'generate notes',
      aliases: ['create notes', 'medical notes', 'soap notes', 'notes'],
      action: () => {
        if (typeof generateMedicalNotes === 'function') generateMedicalNotes()
      },
      feedback: '📝 Generating medical notes...'
    })

    this.registerCommand({
      command: 'save assessment',
      aliases: ['save', 'save results', 'save assessment'],
      action: () => {
        if (typeof saveAssessment === 'function') saveAssessment()
      },
      feedback: '💾 Assessment saved'
    })

    this.registerCommand({
      command: 'next step',
      aliases: ['continue', 'proceed', 'next'],
      action: () => {
        if (typeof nextStep === 'function') nextStep()
      },
      feedback: '➡️ Moving to next step'
    })

    this.registerCommand({
      command: 'previous step',
      aliases: ['go back', 'back', 'last step'],
      action: () => {
        if (typeof previousStep === 'function') previousStep()
      },
      feedback: '⬅️ Going back'
    })

    this.registerCommand({
      command: 'show help',
      aliases: ['help', 'commands', 'what can i say'],
      action: () => {
        this.showHelp()
      },
      feedback: '📖 Showing available commands'
    })

    this.registerCommand({
      command: 'take screenshot',
      aliases: ['screenshot', 'capture', 'photo'],
      action: () => {
        if (typeof takeScreenshot === 'function') takeScreenshot()
      },
      feedback: '📸 Screenshot captured'
    })

    this.registerCommand({
      command: 'start camera',
      aliases: ['open camera', 'enable camera', 'camera on'],
      action: () => {
        if (typeof startCamera === 'function') startCamera()
      },
      feedback: '📷 Camera started'
    })

    this.registerCommand({
      command: 'stop camera',
      aliases: ['close camera', 'disable camera', 'camera off'],
      action: () => {
        if (typeof stopCamera === 'function') stopCamera()
      },
      feedback: '📷 Camera stopped'
    })
  }

  registerCommand(cmd: VoiceAction) {
    this.commands.set(cmd.command, cmd)
  }

  start() {
    if (!this.isSupported || !this.recognition) {
      this.showFeedback('Voice recognition not supported in this browser')
      return false
    }

    try {
      this.recognition.start()
      return true
    } catch (err) {
      console.error('Failed to start recognition:', err)
      return false
    }
  }

  stop() {
    if (this.recognition) {
      this.recognition.stop()
    }
  }

  toggle() {
    if (this.isListening) {
      this.stop()
    } else {
      this.start()
    }
  }

  private showFeedback(message: string) {
    if (this.feedbackElement) {
      this.feedbackElement.textContent = message
      this.feedbackElement.classList.remove('hidden')
      
      setTimeout(() => {
        this.feedbackElement?.classList.add('hidden')
      }, 3000)
    } else {
      // Create toast notification
      this.showToast(message)
    }
  }

  private showToast(message: string) {
    const toast = document.createElement('div')
    toast.className = 'fixed top-4 right-4 bg-gray-900 text-white px-4 py-3 rounded-lg shadow-lg z-50 voice-toast'
    toast.textContent = message
    document.body.appendChild(toast)
    
    setTimeout(() => {
      toast.remove()
    }, 3000)
  }

  private updateStatus(status: string) {
    if (this.statusElement) {
      const icons: Record<string, string> = {
        listening: '🎤',
        idle: '🔇',
        error: '⚠️'
      }
      this.statusElement.textContent = icons[status] || '🔇'
      this.statusElement.className = `voice-status ${status}`
    }
  }

  setFeedbackElement(element: HTMLElement) {
    this.feedbackElement = element
  }

  setStatusElement(element: HTMLElement) {
    this.statusElement = element
  }

  private showHelp() {
    const commands = Array.from(this.commands.values()).map(c => 
      `• ${c.command}`
    ).join('\n')
    
    alert(`Available Voice Commands:\n\n${commands}`)
  }

  getCommands() {
    return Array.from(this.commands.keys())
  }

  getStatus() {
    return {
      isSupported: this.isSupported,
      isListening: this.isListening,
      commandsCount: this.commands.size
    }
  }
}

// Global instance
window.VoiceCommand = new VoiceCommandController()

// Declare global types
declare global {
  interface Window {
    VoiceCommand: VoiceCommandController
    startRecording: () => void
    stopRecording: () => void
    analyzeMovement: () => void
    flipCamera: () => void
    generateMedicalNotes: () => void
    saveAssessment: () => void
    nextStep: () => void
    previousStep: () => void
    takeScreenshot: () => void
    startCamera: () => void
    stopCamera: () => void
  }
}
