var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

// node_modules/unenv/dist/runtime/_internal/utils.mjs
// @__NO_SIDE_EFFECTS__
function createNotImplementedError(name) {
  return new Error(`[unenv] ${name} is not implemented yet!`);
}
__name(createNotImplementedError, "createNotImplementedError");
// @__NO_SIDE_EFFECTS__
function notImplemented(name) {
  const fn = /* @__PURE__ */ __name(() => {
    throw /* @__PURE__ */ createNotImplementedError(name);
  }, "fn");
  return Object.assign(fn, { __unenv__: true });
}
__name(notImplemented, "notImplemented");
// @__NO_SIDE_EFFECTS__
function notImplementedClass(name) {
  return class {
    __unenv__ = true;
    constructor() {
      throw new Error(`[unenv] ${name} is not implemented yet!`);
    }
  };
}
__name(notImplementedClass, "notImplementedClass");

// node_modules/unenv/dist/runtime/node/internal/perf_hooks/performance.mjs
var _timeOrigin = globalThis.performance?.timeOrigin ?? Date.now();
var _performanceNow = globalThis.performance?.now ? globalThis.performance.now.bind(globalThis.performance) : () => Date.now() - _timeOrigin;
var nodeTiming = {
  name: "node",
  entryType: "node",
  startTime: 0,
  duration: 0,
  nodeStart: 0,
  v8Start: 0,
  bootstrapComplete: 0,
  environment: 0,
  loopStart: 0,
  loopExit: 0,
  idleTime: 0,
  uvMetricsInfo: {
    loopCount: 0,
    events: 0,
    eventsWaiting: 0
  },
  detail: void 0,
  toJSON() {
    return this;
  }
};
var PerformanceEntry = class {
  static {
    __name(this, "PerformanceEntry");
  }
  __unenv__ = true;
  detail;
  entryType = "event";
  name;
  startTime;
  constructor(name, options) {
    this.name = name;
    this.startTime = options?.startTime || _performanceNow();
    this.detail = options?.detail;
  }
  get duration() {
    return _performanceNow() - this.startTime;
  }
  toJSON() {
    return {
      name: this.name,
      entryType: this.entryType,
      startTime: this.startTime,
      duration: this.duration,
      detail: this.detail
    };
  }
};
var PerformanceMark = class PerformanceMark2 extends PerformanceEntry {
  static {
    __name(this, "PerformanceMark");
  }
  entryType = "mark";
  constructor() {
    super(...arguments);
  }
  get duration() {
    return 0;
  }
};
var PerformanceMeasure = class extends PerformanceEntry {
  static {
    __name(this, "PerformanceMeasure");
  }
  entryType = "measure";
};
var PerformanceResourceTiming = class extends PerformanceEntry {
  static {
    __name(this, "PerformanceResourceTiming");
  }
  entryType = "resource";
  serverTiming = [];
  connectEnd = 0;
  connectStart = 0;
  decodedBodySize = 0;
  domainLookupEnd = 0;
  domainLookupStart = 0;
  encodedBodySize = 0;
  fetchStart = 0;
  initiatorType = "";
  name = "";
  nextHopProtocol = "";
  redirectEnd = 0;
  redirectStart = 0;
  requestStart = 0;
  responseEnd = 0;
  responseStart = 0;
  secureConnectionStart = 0;
  startTime = 0;
  transferSize = 0;
  workerStart = 0;
  responseStatus = 0;
};
var PerformanceObserverEntryList = class {
  static {
    __name(this, "PerformanceObserverEntryList");
  }
  __unenv__ = true;
  getEntries() {
    return [];
  }
  getEntriesByName(_name, _type) {
    return [];
  }
  getEntriesByType(type) {
    return [];
  }
};
var Performance = class {
  static {
    __name(this, "Performance");
  }
  __unenv__ = true;
  timeOrigin = _timeOrigin;
  eventCounts = /* @__PURE__ */ new Map();
  _entries = [];
  _resourceTimingBufferSize = 0;
  navigation = void 0;
  timing = void 0;
  timerify(_fn, _options) {
    throw createNotImplementedError("Performance.timerify");
  }
  get nodeTiming() {
    return nodeTiming;
  }
  eventLoopUtilization() {
    return {};
  }
  markResourceTiming() {
    return new PerformanceResourceTiming("");
  }
  onresourcetimingbufferfull = null;
  now() {
    if (this.timeOrigin === _timeOrigin) {
      return _performanceNow();
    }
    return Date.now() - this.timeOrigin;
  }
  clearMarks(markName) {
    this._entries = markName ? this._entries.filter((e) => e.name !== markName) : this._entries.filter((e) => e.entryType !== "mark");
  }
  clearMeasures(measureName) {
    this._entries = measureName ? this._entries.filter((e) => e.name !== measureName) : this._entries.filter((e) => e.entryType !== "measure");
  }
  clearResourceTimings() {
    this._entries = this._entries.filter((e) => e.entryType !== "resource" || e.entryType !== "navigation");
  }
  getEntries() {
    return this._entries;
  }
  getEntriesByName(name, type) {
    return this._entries.filter((e) => e.name === name && (!type || e.entryType === type));
  }
  getEntriesByType(type) {
    return this._entries.filter((e) => e.entryType === type);
  }
  mark(name, options) {
    const entry = new PerformanceMark(name, options);
    this._entries.push(entry);
    return entry;
  }
  measure(measureName, startOrMeasureOptions, endMark) {
    let start;
    let end;
    if (typeof startOrMeasureOptions === "string") {
      start = this.getEntriesByName(startOrMeasureOptions, "mark")[0]?.startTime;
      end = this.getEntriesByName(endMark, "mark")[0]?.startTime;
    } else {
      start = Number.parseFloat(startOrMeasureOptions?.start) || this.now();
      end = Number.parseFloat(startOrMeasureOptions?.end) || this.now();
    }
    const entry = new PerformanceMeasure(measureName, {
      startTime: start,
      detail: {
        start,
        end
      }
    });
    this._entries.push(entry);
    return entry;
  }
  setResourceTimingBufferSize(maxSize) {
    this._resourceTimingBufferSize = maxSize;
  }
  addEventListener(type, listener, options) {
    throw createNotImplementedError("Performance.addEventListener");
  }
  removeEventListener(type, listener, options) {
    throw createNotImplementedError("Performance.removeEventListener");
  }
  dispatchEvent(event) {
    throw createNotImplementedError("Performance.dispatchEvent");
  }
  toJSON() {
    return this;
  }
};
var PerformanceObserver = class {
  static {
    __name(this, "PerformanceObserver");
  }
  __unenv__ = true;
  static supportedEntryTypes = [];
  _callback = null;
  constructor(callback) {
    this._callback = callback;
  }
  takeRecords() {
    return [];
  }
  disconnect() {
    throw createNotImplementedError("PerformanceObserver.disconnect");
  }
  observe(options) {
    throw createNotImplementedError("PerformanceObserver.observe");
  }
  bind(fn) {
    return fn;
  }
  runInAsyncScope(fn, thisArg, ...args) {
    return fn.call(thisArg, ...args);
  }
  asyncId() {
    return 0;
  }
  triggerAsyncId() {
    return 0;
  }
  emitDestroy() {
    return this;
  }
};
var performance = globalThis.performance && "addEventListener" in globalThis.performance ? globalThis.performance : new Performance();

// node_modules/@cloudflare/unenv-preset/dist/runtime/polyfill/performance.mjs
globalThis.performance = performance;
globalThis.Performance = Performance;
globalThis.PerformanceEntry = PerformanceEntry;
globalThis.PerformanceMark = PerformanceMark;
globalThis.PerformanceMeasure = PerformanceMeasure;
globalThis.PerformanceObserver = PerformanceObserver;
globalThis.PerformanceObserverEntryList = PerformanceObserverEntryList;
globalThis.PerformanceResourceTiming = PerformanceResourceTiming;

// node_modules/unenv/dist/runtime/node/console.mjs
import { Writable } from "node:stream";

// node_modules/unenv/dist/runtime/mock/noop.mjs
var noop_default = Object.assign(() => {
}, { __unenv__: true });

// node_modules/unenv/dist/runtime/node/console.mjs
var _console = globalThis.console;
var _ignoreErrors = true;
var _stderr = new Writable();
var _stdout = new Writable();
var log = _console?.log ?? noop_default;
var info = _console?.info ?? log;
var trace = _console?.trace ?? info;
var debug = _console?.debug ?? log;
var table = _console?.table ?? log;
var error = _console?.error ?? log;
var warn = _console?.warn ?? error;
var createTask = _console?.createTask ?? /* @__PURE__ */ notImplemented("console.createTask");
var clear = _console?.clear ?? noop_default;
var count = _console?.count ?? noop_default;
var countReset = _console?.countReset ?? noop_default;
var dir = _console?.dir ?? noop_default;
var dirxml = _console?.dirxml ?? noop_default;
var group = _console?.group ?? noop_default;
var groupEnd = _console?.groupEnd ?? noop_default;
var groupCollapsed = _console?.groupCollapsed ?? noop_default;
var profile = _console?.profile ?? noop_default;
var profileEnd = _console?.profileEnd ?? noop_default;
var time = _console?.time ?? noop_default;
var timeEnd = _console?.timeEnd ?? noop_default;
var timeLog = _console?.timeLog ?? noop_default;
var timeStamp = _console?.timeStamp ?? noop_default;
var Console = _console?.Console ?? /* @__PURE__ */ notImplementedClass("console.Console");
var _times = /* @__PURE__ */ new Map();
var _stdoutErrorHandler = noop_default;
var _stderrErrorHandler = noop_default;

// node_modules/@cloudflare/unenv-preset/dist/runtime/node/console.mjs
var workerdConsole = globalThis["console"];
var {
  assert,
  clear: clear2,
  // @ts-expect-error undocumented public API
  context,
  count: count2,
  countReset: countReset2,
  // @ts-expect-error undocumented public API
  createTask: createTask2,
  debug: debug2,
  dir: dir2,
  dirxml: dirxml2,
  error: error2,
  group: group2,
  groupCollapsed: groupCollapsed2,
  groupEnd: groupEnd2,
  info: info2,
  log: log2,
  profile: profile2,
  profileEnd: profileEnd2,
  table: table2,
  time: time2,
  timeEnd: timeEnd2,
  timeLog: timeLog2,
  timeStamp: timeStamp2,
  trace: trace2,
  warn: warn2
} = workerdConsole;
Object.assign(workerdConsole, {
  Console,
  _ignoreErrors,
  _stderr,
  _stderrErrorHandler,
  _stdout,
  _stdoutErrorHandler,
  _times
});
var console_default = workerdConsole;

// node_modules/wrangler/_virtual_unenv_global_polyfill-@cloudflare-unenv-preset-node-console
globalThis.console = console_default;

// node_modules/unenv/dist/runtime/node/internal/process/hrtime.mjs
var hrtime = /* @__PURE__ */ Object.assign(/* @__PURE__ */ __name(function hrtime2(startTime) {
  const now = Date.now();
  const seconds = Math.trunc(now / 1e3);
  const nanos = now % 1e3 * 1e6;
  if (startTime) {
    let diffSeconds = seconds - startTime[0];
    let diffNanos = nanos - startTime[0];
    if (diffNanos < 0) {
      diffSeconds = diffSeconds - 1;
      diffNanos = 1e9 + diffNanos;
    }
    return [diffSeconds, diffNanos];
  }
  return [seconds, nanos];
}, "hrtime"), { bigint: /* @__PURE__ */ __name(function bigint() {
  return BigInt(Date.now() * 1e6);
}, "bigint") });

// node_modules/unenv/dist/runtime/node/internal/process/process.mjs
import { EventEmitter } from "node:events";

// node_modules/unenv/dist/runtime/node/internal/tty/read-stream.mjs
var ReadStream = class {
  static {
    __name(this, "ReadStream");
  }
  fd;
  isRaw = false;
  isTTY = false;
  constructor(fd) {
    this.fd = fd;
  }
  setRawMode(mode) {
    this.isRaw = mode;
    return this;
  }
};

// node_modules/unenv/dist/runtime/node/internal/tty/write-stream.mjs
var WriteStream = class {
  static {
    __name(this, "WriteStream");
  }
  fd;
  columns = 80;
  rows = 24;
  isTTY = false;
  constructor(fd) {
    this.fd = fd;
  }
  clearLine(dir3, callback) {
    callback && callback();
    return false;
  }
  clearScreenDown(callback) {
    callback && callback();
    return false;
  }
  cursorTo(x, y2, callback) {
    callback && typeof callback === "function" && callback();
    return false;
  }
  moveCursor(dx, dy, callback) {
    callback && callback();
    return false;
  }
  getColorDepth(env2) {
    return 1;
  }
  hasColors(count3, env2) {
    return false;
  }
  getWindowSize() {
    return [this.columns, this.rows];
  }
  write(str, encoding, cb) {
    if (str instanceof Uint8Array) {
      str = new TextDecoder().decode(str);
    }
    try {
      console.log(str);
    } catch {
    }
    cb && typeof cb === "function" && cb();
    return false;
  }
};

// node_modules/unenv/dist/runtime/node/internal/process/node-version.mjs
var NODE_VERSION = "22.14.0";

// node_modules/unenv/dist/runtime/node/internal/process/process.mjs
var Process = class _Process extends EventEmitter {
  static {
    __name(this, "Process");
  }
  env;
  hrtime;
  nextTick;
  constructor(impl) {
    super();
    this.env = impl.env;
    this.hrtime = impl.hrtime;
    this.nextTick = impl.nextTick;
    for (const prop of [...Object.getOwnPropertyNames(_Process.prototype), ...Object.getOwnPropertyNames(EventEmitter.prototype)]) {
      const value = this[prop];
      if (typeof value === "function") {
        this[prop] = value.bind(this);
      }
    }
  }
  // --- event emitter ---
  emitWarning(warning, type, code) {
    console.warn(`${code ? `[${code}] ` : ""}${type ? `${type}: ` : ""}${warning}`);
  }
  emit(...args) {
    return super.emit(...args);
  }
  listeners(eventName) {
    return super.listeners(eventName);
  }
  // --- stdio (lazy initializers) ---
  #stdin;
  #stdout;
  #stderr;
  get stdin() {
    return this.#stdin ??= new ReadStream(0);
  }
  get stdout() {
    return this.#stdout ??= new WriteStream(1);
  }
  get stderr() {
    return this.#stderr ??= new WriteStream(2);
  }
  // --- cwd ---
  #cwd = "/";
  chdir(cwd2) {
    this.#cwd = cwd2;
  }
  cwd() {
    return this.#cwd;
  }
  // --- dummy props and getters ---
  arch = "";
  platform = "";
  argv = [];
  argv0 = "";
  execArgv = [];
  execPath = "";
  title = "";
  pid = 200;
  ppid = 100;
  get version() {
    return `v${NODE_VERSION}`;
  }
  get versions() {
    return { node: NODE_VERSION };
  }
  get allowedNodeEnvironmentFlags() {
    return /* @__PURE__ */ new Set();
  }
  get sourceMapsEnabled() {
    return false;
  }
  get debugPort() {
    return 0;
  }
  get throwDeprecation() {
    return false;
  }
  get traceDeprecation() {
    return false;
  }
  get features() {
    return {};
  }
  get release() {
    return {};
  }
  get connected() {
    return false;
  }
  get config() {
    return {};
  }
  get moduleLoadList() {
    return [];
  }
  constrainedMemory() {
    return 0;
  }
  availableMemory() {
    return 0;
  }
  uptime() {
    return 0;
  }
  resourceUsage() {
    return {};
  }
  // --- noop methods ---
  ref() {
  }
  unref() {
  }
  // --- unimplemented methods ---
  umask() {
    throw createNotImplementedError("process.umask");
  }
  getBuiltinModule() {
    return void 0;
  }
  getActiveResourcesInfo() {
    throw createNotImplementedError("process.getActiveResourcesInfo");
  }
  exit() {
    throw createNotImplementedError("process.exit");
  }
  reallyExit() {
    throw createNotImplementedError("process.reallyExit");
  }
  kill() {
    throw createNotImplementedError("process.kill");
  }
  abort() {
    throw createNotImplementedError("process.abort");
  }
  dlopen() {
    throw createNotImplementedError("process.dlopen");
  }
  setSourceMapsEnabled() {
    throw createNotImplementedError("process.setSourceMapsEnabled");
  }
  loadEnvFile() {
    throw createNotImplementedError("process.loadEnvFile");
  }
  disconnect() {
    throw createNotImplementedError("process.disconnect");
  }
  cpuUsage() {
    throw createNotImplementedError("process.cpuUsage");
  }
  setUncaughtExceptionCaptureCallback() {
    throw createNotImplementedError("process.setUncaughtExceptionCaptureCallback");
  }
  hasUncaughtExceptionCaptureCallback() {
    throw createNotImplementedError("process.hasUncaughtExceptionCaptureCallback");
  }
  initgroups() {
    throw createNotImplementedError("process.initgroups");
  }
  openStdin() {
    throw createNotImplementedError("process.openStdin");
  }
  assert() {
    throw createNotImplementedError("process.assert");
  }
  binding() {
    throw createNotImplementedError("process.binding");
  }
  // --- attached interfaces ---
  permission = { has: /* @__PURE__ */ notImplemented("process.permission.has") };
  report = {
    directory: "",
    filename: "",
    signal: "SIGUSR2",
    compact: false,
    reportOnFatalError: false,
    reportOnSignal: false,
    reportOnUncaughtException: false,
    getReport: /* @__PURE__ */ notImplemented("process.report.getReport"),
    writeReport: /* @__PURE__ */ notImplemented("process.report.writeReport")
  };
  finalization = {
    register: /* @__PURE__ */ notImplemented("process.finalization.register"),
    unregister: /* @__PURE__ */ notImplemented("process.finalization.unregister"),
    registerBeforeExit: /* @__PURE__ */ notImplemented("process.finalization.registerBeforeExit")
  };
  memoryUsage = Object.assign(() => ({
    arrayBuffers: 0,
    rss: 0,
    external: 0,
    heapTotal: 0,
    heapUsed: 0
  }), { rss: /* @__PURE__ */ __name(() => 0, "rss") });
  // --- undefined props ---
  mainModule = void 0;
  domain = void 0;
  // optional
  send = void 0;
  exitCode = void 0;
  channel = void 0;
  getegid = void 0;
  geteuid = void 0;
  getgid = void 0;
  getgroups = void 0;
  getuid = void 0;
  setegid = void 0;
  seteuid = void 0;
  setgid = void 0;
  setgroups = void 0;
  setuid = void 0;
  // internals
  _events = void 0;
  _eventsCount = void 0;
  _exiting = void 0;
  _maxListeners = void 0;
  _debugEnd = void 0;
  _debugProcess = void 0;
  _fatalException = void 0;
  _getActiveHandles = void 0;
  _getActiveRequests = void 0;
  _kill = void 0;
  _preload_modules = void 0;
  _rawDebug = void 0;
  _startProfilerIdleNotifier = void 0;
  _stopProfilerIdleNotifier = void 0;
  _tickCallback = void 0;
  _disconnect = void 0;
  _handleQueue = void 0;
  _pendingMessage = void 0;
  _channel = void 0;
  _send = void 0;
  _linkedBinding = void 0;
};

// node_modules/@cloudflare/unenv-preset/dist/runtime/node/process.mjs
var globalProcess = globalThis["process"];
var getBuiltinModule = globalProcess.getBuiltinModule;
var workerdProcess = getBuiltinModule("node:process");
var isWorkerdProcessV2 = globalThis.Cloudflare.compatibilityFlags.enable_nodejs_process_v2;
var unenvProcess = new Process({
  env: globalProcess.env,
  // `hrtime` is only available from workerd process v2
  hrtime: isWorkerdProcessV2 ? workerdProcess.hrtime : hrtime,
  // `nextTick` is available from workerd process v1
  nextTick: workerdProcess.nextTick
});
var { exit, features, platform } = workerdProcess;
var {
  // Always implemented by workerd
  env,
  // Only implemented in workerd v2
  hrtime: hrtime3,
  // Always implemented by workerd
  nextTick
} = unenvProcess;
var {
  _channel,
  _disconnect,
  _events,
  _eventsCount,
  _handleQueue,
  _maxListeners,
  _pendingMessage,
  _send,
  assert: assert2,
  disconnect,
  mainModule
} = unenvProcess;
var {
  // @ts-expect-error `_debugEnd` is missing typings
  _debugEnd,
  // @ts-expect-error `_debugProcess` is missing typings
  _debugProcess,
  // @ts-expect-error `_exiting` is missing typings
  _exiting,
  // @ts-expect-error `_fatalException` is missing typings
  _fatalException,
  // @ts-expect-error `_getActiveHandles` is missing typings
  _getActiveHandles,
  // @ts-expect-error `_getActiveRequests` is missing typings
  _getActiveRequests,
  // @ts-expect-error `_kill` is missing typings
  _kill,
  // @ts-expect-error `_linkedBinding` is missing typings
  _linkedBinding,
  // @ts-expect-error `_preload_modules` is missing typings
  _preload_modules,
  // @ts-expect-error `_rawDebug` is missing typings
  _rawDebug,
  // @ts-expect-error `_startProfilerIdleNotifier` is missing typings
  _startProfilerIdleNotifier,
  // @ts-expect-error `_stopProfilerIdleNotifier` is missing typings
  _stopProfilerIdleNotifier,
  // @ts-expect-error `_tickCallback` is missing typings
  _tickCallback,
  abort,
  addListener,
  allowedNodeEnvironmentFlags,
  arch,
  argv,
  argv0,
  availableMemory,
  // @ts-expect-error `binding` is missing typings
  binding,
  channel,
  chdir,
  config,
  connected,
  constrainedMemory,
  cpuUsage,
  cwd,
  debugPort,
  dlopen,
  // @ts-expect-error `domain` is missing typings
  domain,
  emit,
  emitWarning,
  eventNames,
  execArgv,
  execPath,
  exitCode,
  finalization,
  getActiveResourcesInfo,
  getegid,
  geteuid,
  getgid,
  getgroups,
  getMaxListeners,
  getuid,
  hasUncaughtExceptionCaptureCallback,
  // @ts-expect-error `initgroups` is missing typings
  initgroups,
  kill,
  listenerCount,
  listeners,
  loadEnvFile,
  memoryUsage,
  // @ts-expect-error `moduleLoadList` is missing typings
  moduleLoadList,
  off,
  on,
  once,
  // @ts-expect-error `openStdin` is missing typings
  openStdin,
  permission,
  pid,
  ppid,
  prependListener,
  prependOnceListener,
  rawListeners,
  // @ts-expect-error `reallyExit` is missing typings
  reallyExit,
  ref,
  release,
  removeAllListeners,
  removeListener,
  report,
  resourceUsage,
  send,
  setegid,
  seteuid,
  setgid,
  setgroups,
  setMaxListeners,
  setSourceMapsEnabled,
  setuid,
  setUncaughtExceptionCaptureCallback,
  sourceMapsEnabled,
  stderr,
  stdin,
  stdout,
  throwDeprecation,
  title,
  traceDeprecation,
  umask,
  unref,
  uptime,
  version,
  versions
} = isWorkerdProcessV2 ? workerdProcess : unenvProcess;
var _process = {
  abort,
  addListener,
  allowedNodeEnvironmentFlags,
  hasUncaughtExceptionCaptureCallback,
  setUncaughtExceptionCaptureCallback,
  loadEnvFile,
  sourceMapsEnabled,
  arch,
  argv,
  argv0,
  chdir,
  config,
  connected,
  constrainedMemory,
  availableMemory,
  cpuUsage,
  cwd,
  debugPort,
  dlopen,
  disconnect,
  emit,
  emitWarning,
  env,
  eventNames,
  execArgv,
  execPath,
  exit,
  finalization,
  features,
  getBuiltinModule,
  getActiveResourcesInfo,
  getMaxListeners,
  hrtime: hrtime3,
  kill,
  listeners,
  listenerCount,
  memoryUsage,
  nextTick,
  on,
  off,
  once,
  pid,
  platform,
  ppid,
  prependListener,
  prependOnceListener,
  rawListeners,
  release,
  removeAllListeners,
  removeListener,
  report,
  resourceUsage,
  setMaxListeners,
  setSourceMapsEnabled,
  stderr,
  stdin,
  stdout,
  title,
  throwDeprecation,
  traceDeprecation,
  umask,
  uptime,
  version,
  versions,
  // @ts-expect-error old API
  domain,
  initgroups,
  moduleLoadList,
  reallyExit,
  openStdin,
  assert: assert2,
  binding,
  send,
  exitCode,
  channel,
  getegid,
  geteuid,
  getgid,
  getgroups,
  getuid,
  setegid,
  seteuid,
  setgid,
  setgroups,
  setuid,
  permission,
  mainModule,
  _events,
  _eventsCount,
  _exiting,
  _maxListeners,
  _debugEnd,
  _debugProcess,
  _fatalException,
  _getActiveHandles,
  _getActiveRequests,
  _kill,
  _preload_modules,
  _rawDebug,
  _startProfilerIdleNotifier,
  _stopProfilerIdleNotifier,
  _tickCallback,
  _disconnect,
  _handleQueue,
  _pendingMessage,
  _channel,
  _send,
  _linkedBinding
};
var process_default = _process;

// node_modules/wrangler/_virtual_unenv_global_polyfill-@cloudflare-unenv-preset-node-process
globalThis.process = process_default;

// dist/index.js
var Ct = Object.defineProperty;
var qe = /* @__PURE__ */ __name((t) => {
  throw TypeError(t);
}, "qe");
var Mt = /* @__PURE__ */ __name((t, e, s) => e in t ? Ct(t, e, { enumerable: true, configurable: true, writable: true, value: s }) : t[e] = s, "Mt");
var p = /* @__PURE__ */ __name((t, e, s) => Mt(t, typeof e != "symbol" ? e + "" : e, s), "p");
var Me = /* @__PURE__ */ __name((t, e, s) => e.has(t) || qe("Cannot " + s), "Me");
var c = /* @__PURE__ */ __name((t, e, s) => (Me(t, e, "read from private field"), s ? s.call(t) : e.get(t)), "c");
var g = /* @__PURE__ */ __name((t, e, s) => e.has(t) ? qe("Cannot add the same private member more than once") : e instanceof WeakSet ? e.add(t) : e.set(t, s), "g");
var f = /* @__PURE__ */ __name((t, e, s, r) => (Me(t, e, "write to private field"), r ? r.call(t, s) : e.set(t, s), s), "f");
var y = /* @__PURE__ */ __name((t, e, s) => (Me(t, e, "access private method"), s), "y");
var Be = /* @__PURE__ */ __name((t, e, s, r) => ({ set _(n) {
  f(t, e, n, s);
}, get _() {
  return c(t, e, r);
} }), "Be");
var Ue = /* @__PURE__ */ __name((t, e, s) => (r, n) => {
  let i = -1;
  return a(0);
  async function a(d) {
    if (d <= i) throw new Error("next() called multiple times");
    i = d;
    let o, l = false, u;
    if (t[d] ? (u = t[d][0][0], r.req.routeIndex = d) : u = d === t.length && n || void 0, u) try {
      o = await u(r, () => a(d + 1));
    } catch (h) {
      if (h instanceof Error && e) r.error = h, o = await e(h, r), l = true;
      else throw h;
    }
    else r.finalized === false && s && (o = await s(r));
    return o && (r.finalized === false || l) && (r.res = o), r;
  }
  __name(a, "a");
}, "Ue");
var Nt = Symbol();
var It = /* @__PURE__ */ __name(async (t, e = /* @__PURE__ */ Object.create(null)) => {
  const { all: s = false, dot: r = false } = e, i = (t instanceof dt ? t.raw.headers : t.headers).get("Content-Type");
  return i != null && i.startsWith("multipart/form-data") || i != null && i.startsWith("application/x-www-form-urlencoded") ? Pt(t, { all: s, dot: r }) : {};
}, "It");
async function Pt(t, e) {
  const s = await t.formData();
  return s ? Dt(s, e) : {};
}
__name(Pt, "Pt");
function Dt(t, e) {
  const s = /* @__PURE__ */ Object.create(null);
  return t.forEach((r, n) => {
    e.all || n.endsWith("[]") ? kt(s, n, r) : s[n] = r;
  }), e.dot && Object.entries(s).forEach(([r, n]) => {
    r.includes(".") && (Ht(s, r, n), delete s[r]);
  }), s;
}
__name(Dt, "Dt");
var kt = /* @__PURE__ */ __name((t, e, s) => {
  t[e] !== void 0 ? Array.isArray(t[e]) ? t[e].push(s) : t[e] = [t[e], s] : e.endsWith("[]") ? t[e] = [s] : t[e] = s;
}, "kt");
var Ht = /* @__PURE__ */ __name((t, e, s) => {
  let r = t;
  const n = e.split(".");
  n.forEach((i, a) => {
    a === n.length - 1 ? r[i] = s : ((!r[i] || typeof r[i] != "object" || Array.isArray(r[i]) || r[i] instanceof File) && (r[i] = /* @__PURE__ */ Object.create(null)), r = r[i]);
  });
}, "Ht");
var it = /* @__PURE__ */ __name((t) => {
  const e = t.split("/");
  return e[0] === "" && e.shift(), e;
}, "it");
var Lt = /* @__PURE__ */ __name((t) => {
  const { groups: e, path: s } = Ft(t), r = it(s);
  return $t(r, e);
}, "Lt");
var Ft = /* @__PURE__ */ __name((t) => {
  const e = [];
  return t = t.replace(/\{[^}]+\}/g, (s, r) => {
    const n = `@${r}`;
    return e.push([n, s]), n;
  }), { groups: e, path: t };
}, "Ft");
var $t = /* @__PURE__ */ __name((t, e) => {
  for (let s = e.length - 1; s >= 0; s--) {
    const [r] = e[s];
    for (let n = t.length - 1; n >= 0; n--) if (t[n].includes(r)) {
      t[n] = t[n].replace(r, e[s][1]);
      break;
    }
  }
  return t;
}, "$t");
var Ee = {};
var qt = /* @__PURE__ */ __name((t, e) => {
  if (t === "*") return "*";
  const s = t.match(/^\:([^\{\}]+)(?:\{(.+)\})?$/);
  if (s) {
    const r = `${t}#${e}`;
    return Ee[r] || (s[2] ? Ee[r] = e && e[0] !== ":" && e[0] !== "*" ? [r, s[1], new RegExp(`^${s[2]}(?=/${e})`)] : [t, s[1], new RegExp(`^${s[2]}$`)] : Ee[r] = [t, s[1], true]), Ee[r];
  }
  return null;
}, "qt");
var Fe = /* @__PURE__ */ __name((t, e) => {
  try {
    return e(t);
  } catch {
    return t.replace(/(?:%[0-9A-Fa-f]{2})+/g, (s) => {
      try {
        return e(s);
      } catch {
        return s;
      }
    });
  }
}, "Fe");
var Bt = /* @__PURE__ */ __name((t) => Fe(t, decodeURI), "Bt");
var at = /* @__PURE__ */ __name((t) => {
  const e = t.url, s = e.indexOf("/", e.indexOf(":") + 4);
  let r = s;
  for (; r < e.length; r++) {
    const n = e.charCodeAt(r);
    if (n === 37) {
      const i = e.indexOf("?", r), a = e.slice(s, i === -1 ? void 0 : i);
      return Bt(a.includes("%25") ? a.replace(/%25/g, "%2525") : a);
    } else if (n === 63) break;
  }
  return e.slice(s, r);
}, "at");
var Ut = /* @__PURE__ */ __name((t) => {
  const e = at(t);
  return e.length > 1 && e.at(-1) === "/" ? e.slice(0, -1) : e;
}, "Ut");
var re = /* @__PURE__ */ __name((t, e, ...s) => (s.length && (e = re(e, ...s)), `${(t == null ? void 0 : t[0]) === "/" ? "" : "/"}${t}${e === "/" ? "" : `${(t == null ? void 0 : t.at(-1)) === "/" ? "" : "/"}${(e == null ? void 0 : e[0]) === "/" ? e.slice(1) : e}`}`), "re");
var ot = /* @__PURE__ */ __name((t) => {
  if (t.charCodeAt(t.length - 1) !== 63 || !t.includes(":")) return null;
  const e = t.split("/"), s = [];
  let r = "";
  return e.forEach((n) => {
    if (n !== "" && !/\:/.test(n)) r += "/" + n;
    else if (/\:/.test(n)) if (/\?/.test(n)) {
      s.length === 0 && r === "" ? s.push("/") : s.push(r);
      const i = n.replace("?", "");
      r += "/" + i, s.push(r);
    } else r += "/" + n;
  }), s.filter((n, i, a) => a.indexOf(n) === i);
}, "ot");
var Ne = /* @__PURE__ */ __name((t) => /[%+]/.test(t) ? (t.indexOf("+") !== -1 && (t = t.replace(/\+/g, " ")), t.indexOf("%") !== -1 ? Fe(t, lt) : t) : t, "Ne");
var ct = /* @__PURE__ */ __name((t, e, s) => {
  let r;
  if (!s && e && !/[%+]/.test(e)) {
    let a = t.indexOf(`?${e}`, 8);
    for (a === -1 && (a = t.indexOf(`&${e}`, 8)); a !== -1; ) {
      const d = t.charCodeAt(a + e.length + 1);
      if (d === 61) {
        const o = a + e.length + 2, l = t.indexOf("&", o);
        return Ne(t.slice(o, l === -1 ? void 0 : l));
      } else if (d == 38 || isNaN(d)) return "";
      a = t.indexOf(`&${e}`, a + 1);
    }
    if (r = /[%+]/.test(t), !r) return;
  }
  const n = {};
  r ?? (r = /[%+]/.test(t));
  let i = t.indexOf("?", 8);
  for (; i !== -1; ) {
    const a = t.indexOf("&", i + 1);
    let d = t.indexOf("=", i);
    d > a && a !== -1 && (d = -1);
    let o = t.slice(i + 1, d === -1 ? a === -1 ? void 0 : a : d);
    if (r && (o = Ne(o)), i = a, o === "") continue;
    let l;
    d === -1 ? l = "" : (l = t.slice(d + 1, a === -1 ? void 0 : a), r && (l = Ne(l))), s ? (n[o] && Array.isArray(n[o]) || (n[o] = []), n[o].push(l)) : n[o] ?? (n[o] = l);
  }
  return e ? n[e] : n;
}, "ct");
var zt = ct;
var Wt = /* @__PURE__ */ __name((t, e) => ct(t, e, true), "Wt");
var lt = decodeURIComponent;
var ze = /* @__PURE__ */ __name((t) => Fe(t, lt), "ze");
var ae;
var A;
var $;
var ut;
var ht;
var ke;
var B;
var Ye;
var dt = (Ye = class {
  static {
    __name(this, "Ye");
  }
  constructor(t, e = "/", s = [[]]) {
    g(this, $);
    p(this, "raw");
    g(this, ae);
    g(this, A);
    p(this, "routeIndex", 0);
    p(this, "path");
    p(this, "bodyCache", {});
    g(this, B, (t2) => {
      const { bodyCache: e2, raw: s2 } = this, r = e2[t2];
      if (r) return r;
      const n = Object.keys(e2)[0];
      return n ? e2[n].then((i) => (n === "json" && (i = JSON.stringify(i)), new Response(i)[t2]())) : e2[t2] = s2[t2]();
    });
    this.raw = t, this.path = e, f(this, A, s), f(this, ae, {});
  }
  param(t) {
    return t ? y(this, $, ut).call(this, t) : y(this, $, ht).call(this);
  }
  query(t) {
    return zt(this.url, t);
  }
  queries(t) {
    return Wt(this.url, t);
  }
  header(t) {
    if (t) return this.raw.headers.get(t) ?? void 0;
    const e = {};
    return this.raw.headers.forEach((s, r) => {
      e[r] = s;
    }), e;
  }
  async parseBody(t) {
    var e;
    return (e = this.bodyCache).parsedBody ?? (e.parsedBody = await It(this, t));
  }
  json() {
    return c(this, B).call(this, "text").then((t) => JSON.parse(t));
  }
  text() {
    return c(this, B).call(this, "text");
  }
  arrayBuffer() {
    return c(this, B).call(this, "arrayBuffer");
  }
  blob() {
    return c(this, B).call(this, "blob");
  }
  formData() {
    return c(this, B).call(this, "formData");
  }
  addValidatedData(t, e) {
    c(this, ae)[t] = e;
  }
  valid(t) {
    return c(this, ae)[t];
  }
  get url() {
    return this.raw.url;
  }
  get method() {
    return this.raw.method;
  }
  get [Nt]() {
    return c(this, A);
  }
  get matchedRoutes() {
    return c(this, A)[0].map(([[, t]]) => t);
  }
  get routePath() {
    return c(this, A)[0].map(([[, t]]) => t)[this.routeIndex].path;
  }
}, ae = /* @__PURE__ */ new WeakMap(), A = /* @__PURE__ */ new WeakMap(), $ = /* @__PURE__ */ new WeakSet(), ut = /* @__PURE__ */ __name(function(t) {
  const e = c(this, A)[0][this.routeIndex][1][t], s = y(this, $, ke).call(this, e);
  return s && /\%/.test(s) ? ze(s) : s;
}, "ut"), ht = /* @__PURE__ */ __name(function() {
  const t = {}, e = Object.keys(c(this, A)[0][this.routeIndex][1]);
  for (const s of e) {
    const r = y(this, $, ke).call(this, c(this, A)[0][this.routeIndex][1][s]);
    r !== void 0 && (t[s] = /\%/.test(r) ? ze(r) : r);
  }
  return t;
}, "ht"), ke = /* @__PURE__ */ __name(function(t) {
  return c(this, A)[1] ? c(this, A)[1][t] : t;
}, "ke"), B = /* @__PURE__ */ new WeakMap(), Ye);
var Kt = { Stringify: 1 };
var ft = /* @__PURE__ */ __name(async (t, e, s, r, n) => {
  typeof t == "object" && !(t instanceof String) && (t instanceof Promise || (t = t.toString()), t instanceof Promise && (t = await t));
  const i = t.callbacks;
  return i != null && i.length ? (n ? n[0] += t : n = [t], Promise.all(i.map((d) => d({ phase: e, buffer: n, context: r }))).then((d) => Promise.all(d.filter(Boolean).map((o) => ft(o, e, false, r, n))).then(() => n[0]))) : Promise.resolve(t);
}, "ft");
var Vt = "text/plain; charset=UTF-8";
var Ie = /* @__PURE__ */ __name((t, e) => ({ "Content-Type": t, ...e }), "Ie");
var ge;
var _e;
var k;
var oe;
var H;
var T;
var ye;
var ce;
var le;
var Y;
var xe;
var ve;
var U;
var ne;
var Xe;
var Jt = (Xe = class {
  static {
    __name(this, "Xe");
  }
  constructor(t, e) {
    g(this, U);
    g(this, ge);
    g(this, _e);
    p(this, "env", {});
    g(this, k);
    p(this, "finalized", false);
    p(this, "error");
    g(this, oe);
    g(this, H);
    g(this, T);
    g(this, ye);
    g(this, ce);
    g(this, le);
    g(this, Y);
    g(this, xe);
    g(this, ve);
    p(this, "render", (...t2) => (c(this, ce) ?? f(this, ce, (e2) => this.html(e2)), c(this, ce).call(this, ...t2)));
    p(this, "setLayout", (t2) => f(this, ye, t2));
    p(this, "getLayout", () => c(this, ye));
    p(this, "setRenderer", (t2) => {
      f(this, ce, t2);
    });
    p(this, "header", (t2, e2, s) => {
      this.finalized && f(this, T, new Response(c(this, T).body, c(this, T)));
      const r = c(this, T) ? c(this, T).headers : c(this, Y) ?? f(this, Y, new Headers());
      e2 === void 0 ? r.delete(t2) : s != null && s.append ? r.append(t2, e2) : r.set(t2, e2);
    });
    p(this, "status", (t2) => {
      f(this, oe, t2);
    });
    p(this, "set", (t2, e2) => {
      c(this, k) ?? f(this, k, /* @__PURE__ */ new Map()), c(this, k).set(t2, e2);
    });
    p(this, "get", (t2) => c(this, k) ? c(this, k).get(t2) : void 0);
    p(this, "newResponse", (...t2) => y(this, U, ne).call(this, ...t2));
    p(this, "body", (t2, e2, s) => y(this, U, ne).call(this, t2, e2, s));
    p(this, "text", (t2, e2, s) => !c(this, Y) && !c(this, oe) && !e2 && !s && !this.finalized ? new Response(t2) : y(this, U, ne).call(this, t2, e2, Ie(Vt, s)));
    p(this, "json", (t2, e2, s) => y(this, U, ne).call(this, JSON.stringify(t2), e2, Ie("application/json", s)));
    p(this, "html", (t2, e2, s) => {
      const r = /* @__PURE__ */ __name((n) => y(this, U, ne).call(this, n, e2, Ie("text/html; charset=UTF-8", s)), "r");
      return typeof t2 == "object" ? ft(t2, Kt.Stringify, false, {}).then(r) : r(t2);
    });
    p(this, "redirect", (t2, e2) => {
      const s = String(t2);
      return this.header("Location", /[^\x00-\xFF]/.test(s) ? encodeURI(s) : s), this.newResponse(null, e2 ?? 302);
    });
    p(this, "notFound", () => (c(this, le) ?? f(this, le, () => new Response()), c(this, le).call(this, this)));
    f(this, ge, t), e && (f(this, H, e.executionCtx), this.env = e.env, f(this, le, e.notFoundHandler), f(this, ve, e.path), f(this, xe, e.matchResult));
  }
  get req() {
    return c(this, _e) ?? f(this, _e, new dt(c(this, ge), c(this, ve), c(this, xe))), c(this, _e);
  }
  get event() {
    if (c(this, H) && "respondWith" in c(this, H)) return c(this, H);
    throw Error("This context has no FetchEvent");
  }
  get executionCtx() {
    if (c(this, H)) return c(this, H);
    throw Error("This context has no ExecutionContext");
  }
  get res() {
    return c(this, T) || f(this, T, new Response(null, { headers: c(this, Y) ?? f(this, Y, new Headers()) }));
  }
  set res(t) {
    if (c(this, T) && t) {
      t = new Response(t.body, t);
      for (const [e, s] of c(this, T).headers.entries()) if (e !== "content-type") if (e === "set-cookie") {
        const r = c(this, T).headers.getSetCookie();
        t.headers.delete("set-cookie");
        for (const n of r) t.headers.append("set-cookie", n);
      } else t.headers.set(e, s);
    }
    f(this, T, t), this.finalized = true;
  }
  get var() {
    return c(this, k) ? Object.fromEntries(c(this, k)) : {};
  }
}, ge = /* @__PURE__ */ new WeakMap(), _e = /* @__PURE__ */ new WeakMap(), k = /* @__PURE__ */ new WeakMap(), oe = /* @__PURE__ */ new WeakMap(), H = /* @__PURE__ */ new WeakMap(), T = /* @__PURE__ */ new WeakMap(), ye = /* @__PURE__ */ new WeakMap(), ce = /* @__PURE__ */ new WeakMap(), le = /* @__PURE__ */ new WeakMap(), Y = /* @__PURE__ */ new WeakMap(), xe = /* @__PURE__ */ new WeakMap(), ve = /* @__PURE__ */ new WeakMap(), U = /* @__PURE__ */ new WeakSet(), ne = /* @__PURE__ */ __name(function(t, e, s) {
  const r = c(this, T) ? new Headers(c(this, T).headers) : c(this, Y) ?? new Headers();
  if (typeof e == "object" && "headers" in e) {
    const i = e.headers instanceof Headers ? e.headers : new Headers(e.headers);
    for (const [a, d] of i) a.toLowerCase() === "set-cookie" ? r.append(a, d) : r.set(a, d);
  }
  if (s) for (const [i, a] of Object.entries(s)) if (typeof a == "string") r.set(i, a);
  else {
    r.delete(i);
    for (const d of a) r.append(i, d);
  }
  const n = typeof e == "number" ? e : (e == null ? void 0 : e.status) ?? c(this, oe);
  return new Response(t, { status: n, headers: r });
}, "ne"), Xe);
var E = "ALL";
var Gt = "all";
var Yt = ["get", "post", "put", "delete", "options", "patch"];
var pt = "Can not add a route since the matcher is already built.";
var mt = class extends Error {
  static {
    __name(this, "mt");
  }
};
var Xt = "__COMPOSED_HANDLER";
var Qt = /* @__PURE__ */ __name((t) => t.text("404 Not Found", 404), "Qt");
var We = /* @__PURE__ */ __name((t, e) => {
  if ("getResponse" in t) {
    const s = t.getResponse();
    return e.newResponse(s.body, s);
  }
  return console.error(t), e.text("Internal Server Error", 500);
}, "We");
var C;
var b;
var _t;
var M;
var J;
var be;
var Re;
var Qe;
var gt = (Qe = class {
  static {
    __name(this, "Qe");
  }
  constructor(e = {}) {
    g(this, b);
    p(this, "get");
    p(this, "post");
    p(this, "put");
    p(this, "delete");
    p(this, "options");
    p(this, "patch");
    p(this, "all");
    p(this, "on");
    p(this, "use");
    p(this, "router");
    p(this, "getPath");
    p(this, "_basePath", "/");
    g(this, C, "/");
    p(this, "routes", []);
    g(this, M, Qt);
    p(this, "errorHandler", We);
    p(this, "onError", (e2) => (this.errorHandler = e2, this));
    p(this, "notFound", (e2) => (f(this, M, e2), this));
    p(this, "fetch", (e2, ...s) => y(this, b, Re).call(this, e2, s[1], s[0], e2.method));
    p(this, "request", (e2, s, r2, n2) => e2 instanceof Request ? this.fetch(s ? new Request(e2, s) : e2, r2, n2) : (e2 = e2.toString(), this.fetch(new Request(/^https?:\/\//.test(e2) ? e2 : `http://localhost${re("/", e2)}`, s), r2, n2)));
    p(this, "fire", () => {
      addEventListener("fetch", (e2) => {
        e2.respondWith(y(this, b, Re).call(this, e2.request, e2, void 0, e2.request.method));
      });
    });
    [...Yt, Gt].forEach((i) => {
      this[i] = (a, ...d) => (typeof a == "string" ? f(this, C, a) : y(this, b, J).call(this, i, c(this, C), a), d.forEach((o) => {
        y(this, b, J).call(this, i, c(this, C), o);
      }), this);
    }), this.on = (i, a, ...d) => {
      for (const o of [a].flat()) {
        f(this, C, o);
        for (const l of [i].flat()) d.map((u) => {
          y(this, b, J).call(this, l.toUpperCase(), c(this, C), u);
        });
      }
      return this;
    }, this.use = (i, ...a) => (typeof i == "string" ? f(this, C, i) : (f(this, C, "*"), a.unshift(i)), a.forEach((d) => {
      y(this, b, J).call(this, E, c(this, C), d);
    }), this);
    const { strict: r, ...n } = e;
    Object.assign(this, n), this.getPath = r ?? true ? e.getPath ?? at : Ut;
  }
  route(e, s) {
    const r = this.basePath(e);
    return s.routes.map((n) => {
      var a;
      let i;
      s.errorHandler === We ? i = n.handler : (i = /* @__PURE__ */ __name(async (d, o) => (await Ue([], s.errorHandler)(d, () => n.handler(d, o))).res, "i"), i[Xt] = n.handler), y(a = r, b, J).call(a, n.method, n.path, i);
    }), this;
  }
  basePath(e) {
    const s = y(this, b, _t).call(this);
    return s._basePath = re(this._basePath, e), s;
  }
  mount(e, s, r) {
    let n, i;
    r && (typeof r == "function" ? i = r : (i = r.optionHandler, r.replaceRequest === false ? n = /* @__PURE__ */ __name((o) => o, "n") : n = r.replaceRequest));
    const a = i ? (o) => {
      const l = i(o);
      return Array.isArray(l) ? l : [l];
    } : (o) => {
      let l;
      try {
        l = o.executionCtx;
      } catch {
      }
      return [o.env, l];
    };
    n || (n = (() => {
      const o = re(this._basePath, e), l = o === "/" ? 0 : o.length;
      return (u) => {
        const h = new URL(u.url);
        return h.pathname = h.pathname.slice(l) || "/", new Request(h, u);
      };
    })());
    const d = /* @__PURE__ */ __name(async (o, l) => {
      const u = await s(n(o.req.raw), ...a(o));
      if (u) return u;
      await l();
    }, "d");
    return y(this, b, J).call(this, E, re(e, "*"), d), this;
  }
}, C = /* @__PURE__ */ new WeakMap(), b = /* @__PURE__ */ new WeakSet(), _t = /* @__PURE__ */ __name(function() {
  const e = new gt({ router: this.router, getPath: this.getPath });
  return e.errorHandler = this.errorHandler, f(e, M, c(this, M)), e.routes = this.routes, e;
}, "_t"), M = /* @__PURE__ */ new WeakMap(), J = /* @__PURE__ */ __name(function(e, s, r) {
  e = e.toUpperCase(), s = re(this._basePath, s);
  const n = { basePath: this._basePath, path: s, method: e, handler: r };
  this.router.add(e, s, [r, n]), this.routes.push(n);
}, "J"), be = /* @__PURE__ */ __name(function(e, s) {
  if (e instanceof Error) return this.errorHandler(e, s);
  throw e;
}, "be"), Re = /* @__PURE__ */ __name(function(e, s, r, n) {
  if (n === "HEAD") return (async () => new Response(null, await y(this, b, Re).call(this, e, s, r, "GET")))();
  const i = this.getPath(e, { env: r }), a = this.router.match(n, i), d = new Jt(e, { path: i, matchResult: a, env: r, executionCtx: s, notFoundHandler: c(this, M) });
  if (a[0].length === 1) {
    let l;
    try {
      l = a[0][0][0][0](d, async () => {
        d.res = await c(this, M).call(this, d);
      });
    } catch (u) {
      return y(this, b, be).call(this, u, d);
    }
    return l instanceof Promise ? l.then((u) => u || (d.finalized ? d.res : c(this, M).call(this, d))).catch((u) => y(this, b, be).call(this, u, d)) : l ?? c(this, M).call(this, d);
  }
  const o = Ue(a[0], this.errorHandler, c(this, M));
  return (async () => {
    try {
      const l = await o(d);
      if (!l.finalized) throw new Error("Context is not finalized. Did you forget to return a Response object or `await next()`?");
      return l.res;
    } catch (l) {
      return y(this, b, be).call(this, l, d);
    }
  })();
}, "Re"), Qe);
var yt = [];
function Zt(t, e) {
  const s = this.buildAllMatchers(), r = /* @__PURE__ */ __name((n, i) => {
    const a = s[n] || s[E], d = a[2][i];
    if (d) return d;
    const o = i.match(a[0]);
    if (!o) return [[], yt];
    const l = o.indexOf("", 1);
    return [a[1][l], o];
  }, "r");
  return this.match = r, r(t, e);
}
__name(Zt, "Zt");
var Se = "[^/]+";
var pe = ".*";
var me = "(?:|/.*)";
var ie = Symbol();
var es = new Set(".\\+*[^]$()");
function ts(t, e) {
  return t.length === 1 ? e.length === 1 ? t < e ? -1 : 1 : -1 : e.length === 1 || t === pe || t === me ? 1 : e === pe || e === me ? -1 : t === Se ? 1 : e === Se ? -1 : t.length === e.length ? t < e ? -1 : 1 : e.length - t.length;
}
__name(ts, "ts");
var X;
var Q;
var N;
var Ze;
var He = (Ze = class {
  static {
    __name(this, "Ze");
  }
  constructor() {
    g(this, X);
    g(this, Q);
    g(this, N, /* @__PURE__ */ Object.create(null));
  }
  insert(e, s, r, n, i) {
    if (e.length === 0) {
      if (c(this, X) !== void 0) throw ie;
      if (i) return;
      f(this, X, s);
      return;
    }
    const [a, ...d] = e, o = a === "*" ? d.length === 0 ? ["", "", pe] : ["", "", Se] : a === "/*" ? ["", "", me] : a.match(/^\:([^\{\}]+)(?:\{(.+)\})?$/);
    let l;
    if (o) {
      const u = o[1];
      let h = o[2] || Se;
      if (u && o[2] && (h === ".*" || (h = h.replace(/^\((?!\?:)(?=[^)]+\)$)/, "(?:"), /\((?!\?:)/.test(h)))) throw ie;
      if (l = c(this, N)[h], !l) {
        if (Object.keys(c(this, N)).some((m) => m !== pe && m !== me)) throw ie;
        if (i) return;
        l = c(this, N)[h] = new He(), u !== "" && f(l, Q, n.varIndex++);
      }
      !i && u !== "" && r.push([u, c(l, Q)]);
    } else if (l = c(this, N)[a], !l) {
      if (Object.keys(c(this, N)).some((u) => u.length > 1 && u !== pe && u !== me)) throw ie;
      if (i) return;
      l = c(this, N)[a] = new He();
    }
    l.insert(d, s, r, n, i);
  }
  buildRegExpStr() {
    const s = Object.keys(c(this, N)).sort(ts).map((r) => {
      const n = c(this, N)[r];
      return (typeof c(n, Q) == "number" ? `(${r})@${c(n, Q)}` : es.has(r) ? `\\${r}` : r) + n.buildRegExpStr();
    });
    return typeof c(this, X) == "number" && s.unshift(`#${c(this, X)}`), s.length === 0 ? "" : s.length === 1 ? s[0] : "(?:" + s.join("|") + ")";
  }
}, X = /* @__PURE__ */ new WeakMap(), Q = /* @__PURE__ */ new WeakMap(), N = /* @__PURE__ */ new WeakMap(), Ze);
var Te;
var we;
var et;
var ss = (et = class {
  static {
    __name(this, "et");
  }
  constructor() {
    g(this, Te, { varIndex: 0 });
    g(this, we, new He());
  }
  insert(t, e, s) {
    const r = [], n = [];
    for (let a = 0; ; ) {
      let d = false;
      if (t = t.replace(/\{[^}]+\}/g, (o) => {
        const l = `@\\${a}`;
        return n[a] = [l, o], a++, d = true, l;
      }), !d) break;
    }
    const i = t.match(/(?::[^\/]+)|(?:\/\*$)|./g) || [];
    for (let a = n.length - 1; a >= 0; a--) {
      const [d] = n[a];
      for (let o = i.length - 1; o >= 0; o--) if (i[o].indexOf(d) !== -1) {
        i[o] = i[o].replace(d, n[a][1]);
        break;
      }
    }
    return c(this, we).insert(i, e, r, c(this, Te), s), r;
  }
  buildRegExp() {
    let t = c(this, we).buildRegExpStr();
    if (t === "") return [/^$/, [], []];
    let e = 0;
    const s = [], r = [];
    return t = t.replace(/#(\d+)|@(\d+)|\.\*\$/g, (n, i, a) => i !== void 0 ? (s[++e] = Number(i), "$()") : (a !== void 0 && (r[Number(a)] = ++e), "")), [new RegExp(`^${t}`), s, r];
  }
}, Te = /* @__PURE__ */ new WeakMap(), we = /* @__PURE__ */ new WeakMap(), et);
var rs = [/^$/, [], /* @__PURE__ */ Object.create(null)];
var je = /* @__PURE__ */ Object.create(null);
function xt(t) {
  return je[t] ?? (je[t] = new RegExp(t === "*" ? "" : `^${t.replace(/\/\*$|([.\\+*[^\]$()])/g, (e, s) => s ? `\\${s}` : "(?:|/.*)")}$`));
}
__name(xt, "xt");
function ns() {
  je = /* @__PURE__ */ Object.create(null);
}
__name(ns, "ns");
function is(t) {
  var l;
  const e = new ss(), s = [];
  if (t.length === 0) return rs;
  const r = t.map((u) => [!/\*|\/:/.test(u[0]), ...u]).sort(([u, h], [m, v]) => u ? 1 : m ? -1 : h.length - v.length), n = /* @__PURE__ */ Object.create(null);
  for (let u = 0, h = -1, m = r.length; u < m; u++) {
    const [v, O, x] = r[u];
    v ? n[O] = [x.map(([S]) => [S, /* @__PURE__ */ Object.create(null)]), yt] : h++;
    let w;
    try {
      w = e.insert(O, h, v);
    } catch (S) {
      throw S === ie ? new mt(O) : S;
    }
    v || (s[h] = x.map(([S, te]) => {
      const ue = /* @__PURE__ */ Object.create(null);
      for (te -= 1; te >= 0; te--) {
        const [I, Ae] = w[te];
        ue[I] = Ae;
      }
      return [S, ue];
    }));
  }
  const [i, a, d] = e.buildRegExp();
  for (let u = 0, h = s.length; u < h; u++) for (let m = 0, v = s[u].length; m < v; m++) {
    const O = (l = s[u][m]) == null ? void 0 : l[1];
    if (!O) continue;
    const x = Object.keys(O);
    for (let w = 0, S = x.length; w < S; w++) O[x[w]] = d[O[x[w]]];
  }
  const o = [];
  for (const u in a) o[u] = s[a[u]];
  return [i, o, n];
}
__name(is, "is");
function se(t, e) {
  if (t) {
    for (const s of Object.keys(t).sort((r, n) => n.length - r.length)) if (xt(s).test(e)) return [...t[s]];
  }
}
__name(se, "se");
var z;
var W;
var Oe;
var vt;
var tt;
var as = (tt = class {
  static {
    __name(this, "tt");
  }
  constructor() {
    g(this, Oe);
    p(this, "name", "RegExpRouter");
    g(this, z);
    g(this, W);
    p(this, "match", Zt);
    f(this, z, { [E]: /* @__PURE__ */ Object.create(null) }), f(this, W, { [E]: /* @__PURE__ */ Object.create(null) });
  }
  add(t, e, s) {
    var d;
    const r = c(this, z), n = c(this, W);
    if (!r || !n) throw new Error(pt);
    r[t] || [r, n].forEach((o) => {
      o[t] = /* @__PURE__ */ Object.create(null), Object.keys(o[E]).forEach((l) => {
        o[t][l] = [...o[E][l]];
      });
    }), e === "/*" && (e = "*");
    const i = (e.match(/\/:/g) || []).length;
    if (/\*$/.test(e)) {
      const o = xt(e);
      t === E ? Object.keys(r).forEach((l) => {
        var u;
        (u = r[l])[e] || (u[e] = se(r[l], e) || se(r[E], e) || []);
      }) : (d = r[t])[e] || (d[e] = se(r[t], e) || se(r[E], e) || []), Object.keys(r).forEach((l) => {
        (t === E || t === l) && Object.keys(r[l]).forEach((u) => {
          o.test(u) && r[l][u].push([s, i]);
        });
      }), Object.keys(n).forEach((l) => {
        (t === E || t === l) && Object.keys(n[l]).forEach((u) => o.test(u) && n[l][u].push([s, i]));
      });
      return;
    }
    const a = ot(e) || [e];
    for (let o = 0, l = a.length; o < l; o++) {
      const u = a[o];
      Object.keys(n).forEach((h) => {
        var m;
        (t === E || t === h) && ((m = n[h])[u] || (m[u] = [...se(r[h], u) || se(r[E], u) || []]), n[h][u].push([s, i - l + o + 1]));
      });
    }
  }
  buildAllMatchers() {
    const t = /* @__PURE__ */ Object.create(null);
    return Object.keys(c(this, W)).concat(Object.keys(c(this, z))).forEach((e) => {
      t[e] || (t[e] = y(this, Oe, vt).call(this, e));
    }), f(this, z, f(this, W, void 0)), ns(), t;
  }
}, z = /* @__PURE__ */ new WeakMap(), W = /* @__PURE__ */ new WeakMap(), Oe = /* @__PURE__ */ new WeakSet(), vt = /* @__PURE__ */ __name(function(t) {
  const e = [];
  let s = t === E;
  return [c(this, z), c(this, W)].forEach((r) => {
    const n = r[t] ? Object.keys(r[t]).map((i) => [i, r[t][i]]) : [];
    n.length !== 0 ? (s || (s = true), e.push(...n)) : t !== E && e.push(...Object.keys(r[E]).map((i) => [i, r[E][i]]));
  }), s ? is(e) : null;
}, "vt"), tt);
var K;
var L;
var st;
var os = (st = class {
  static {
    __name(this, "st");
  }
  constructor(t) {
    p(this, "name", "SmartRouter");
    g(this, K, []);
    g(this, L, []);
    f(this, K, t.routers);
  }
  add(t, e, s) {
    if (!c(this, L)) throw new Error(pt);
    c(this, L).push([t, e, s]);
  }
  match(t, e) {
    if (!c(this, L)) throw new Error("Fatal error");
    const s = c(this, K), r = c(this, L), n = s.length;
    let i = 0, a;
    for (; i < n; i++) {
      const d = s[i];
      try {
        for (let o = 0, l = r.length; o < l; o++) d.add(...r[o]);
        a = d.match(t, e);
      } catch (o) {
        if (o instanceof mt) continue;
        throw o;
      }
      this.match = d.match.bind(d), f(this, K, [d]), f(this, L, void 0);
      break;
    }
    if (i === n) throw new Error("Fatal error");
    return this.name = `SmartRouter + ${this.activeRouter.name}`, a;
  }
  get activeRouter() {
    if (c(this, L) || c(this, K).length !== 1) throw new Error("No active router has been determined yet.");
    return c(this, K)[0];
  }
}, K = /* @__PURE__ */ new WeakMap(), L = /* @__PURE__ */ new WeakMap(), st);
var fe = /* @__PURE__ */ Object.create(null);
var V;
var j;
var Z;
var de;
var R;
var F;
var G;
var rt;
var wt = (rt = class {
  static {
    __name(this, "rt");
  }
  constructor(t, e, s) {
    g(this, F);
    g(this, V);
    g(this, j);
    g(this, Z);
    g(this, de, 0);
    g(this, R, fe);
    if (f(this, j, s || /* @__PURE__ */ Object.create(null)), f(this, V, []), t && e) {
      const r = /* @__PURE__ */ Object.create(null);
      r[t] = { handler: e, possibleKeys: [], score: 0 }, f(this, V, [r]);
    }
    f(this, Z, []);
  }
  insert(t, e, s) {
    f(this, de, ++Be(this, de)._);
    let r = this;
    const n = Lt(e), i = [];
    for (let a = 0, d = n.length; a < d; a++) {
      const o = n[a], l = n[a + 1], u = qt(o, l), h = Array.isArray(u) ? u[0] : o;
      if (h in c(r, j)) {
        r = c(r, j)[h], u && i.push(u[1]);
        continue;
      }
      c(r, j)[h] = new wt(), u && (c(r, Z).push(u), i.push(u[1])), r = c(r, j)[h];
    }
    return c(r, V).push({ [t]: { handler: s, possibleKeys: i.filter((a, d, o) => o.indexOf(a) === d), score: c(this, de) } }), r;
  }
  search(t, e) {
    var d;
    const s = [];
    f(this, R, fe);
    let n = [this];
    const i = it(e), a = [];
    for (let o = 0, l = i.length; o < l; o++) {
      const u = i[o], h = o === l - 1, m = [];
      for (let v = 0, O = n.length; v < O; v++) {
        const x = n[v], w = c(x, j)[u];
        w && (f(w, R, c(x, R)), h ? (c(w, j)["*"] && s.push(...y(this, F, G).call(this, c(w, j)["*"], t, c(x, R))), s.push(...y(this, F, G).call(this, w, t, c(x, R)))) : m.push(w));
        for (let S = 0, te = c(x, Z).length; S < te; S++) {
          const ue = c(x, Z)[S], I = c(x, R) === fe ? {} : { ...c(x, R) };
          if (ue === "*") {
            const q = c(x, j)["*"];
            q && (s.push(...y(this, F, G).call(this, q, t, c(x, R))), f(q, R, I), m.push(q));
            continue;
          }
          const [Ae, $e, he] = ue;
          if (!u && !(he instanceof RegExp)) continue;
          const D = c(x, j)[Ae], At = i.slice(o).join("/");
          if (he instanceof RegExp) {
            const q = he.exec(At);
            if (q) {
              if (I[$e] = q[0], s.push(...y(this, F, G).call(this, D, t, c(x, R), I)), Object.keys(c(D, j)).length) {
                f(D, R, I);
                const Ce = ((d = q[0].match(/\//)) == null ? void 0 : d.length) ?? 0;
                (a[Ce] || (a[Ce] = [])).push(D);
              }
              continue;
            }
          }
          (he === true || he.test(u)) && (I[$e] = u, h ? (s.push(...y(this, F, G).call(this, D, t, I, c(x, R))), c(D, j)["*"] && s.push(...y(this, F, G).call(this, c(D, j)["*"], t, I, c(x, R)))) : (f(D, R, I), m.push(D)));
        }
      }
      n = m.concat(a.shift() ?? []);
    }
    return s.length > 1 && s.sort((o, l) => o.score - l.score), [s.map(({ handler: o, params: l }) => [o, l])];
  }
}, V = /* @__PURE__ */ new WeakMap(), j = /* @__PURE__ */ new WeakMap(), Z = /* @__PURE__ */ new WeakMap(), de = /* @__PURE__ */ new WeakMap(), R = /* @__PURE__ */ new WeakMap(), F = /* @__PURE__ */ new WeakSet(), G = /* @__PURE__ */ __name(function(t, e, s, r) {
  const n = [];
  for (let i = 0, a = c(t, V).length; i < a; i++) {
    const d = c(t, V)[i], o = d[e] || d[E], l = {};
    if (o !== void 0 && (o.params = /* @__PURE__ */ Object.create(null), n.push(o), s !== fe || r && r !== fe)) for (let u = 0, h = o.possibleKeys.length; u < h; u++) {
      const m = o.possibleKeys[u], v = l[o.score];
      o.params[m] = r != null && r[m] && !v ? r[m] : s[m] ?? (r == null ? void 0 : r[m]), l[o.score] = true;
    }
  }
  return n;
}, "G"), rt);
var ee;
var nt;
var cs = (nt = class {
  static {
    __name(this, "nt");
  }
  constructor() {
    p(this, "name", "TrieRouter");
    g(this, ee);
    f(this, ee, new wt());
  }
  add(t, e, s) {
    const r = ot(e);
    if (r) {
      for (let n = 0, i = r.length; n < i; n++) c(this, ee).insert(t, r[n], s);
      return;
    }
    c(this, ee).insert(t, e, s);
  }
  match(t, e) {
    return c(this, ee).search(t, e);
  }
}, ee = /* @__PURE__ */ new WeakMap(), nt);
var Et = class extends gt {
  static {
    __name(this, "Et");
  }
  constructor(t = {}) {
    super(t), this.router = t.router ?? new os({ routers: [new as(), new cs()] });
  }
};
var ls = /* @__PURE__ */ __name((t) => {
  const s = { ...{ origin: "*", allowMethods: ["GET", "HEAD", "PUT", "POST", "DELETE", "PATCH"], allowHeaders: [], exposeHeaders: [] }, ...t }, r = /* @__PURE__ */ ((i) => typeof i == "string" ? i === "*" ? () => i : (a) => i === a ? a : null : typeof i == "function" ? i : (a) => i.includes(a) ? a : null)(s.origin), n = ((i) => typeof i == "function" ? i : Array.isArray(i) ? () => i : () => [])(s.allowMethods);
  return async function(a, d) {
    var u;
    function o(h, m) {
      a.res.headers.set(h, m);
    }
    __name(o, "o");
    const l = await r(a.req.header("origin") || "", a);
    if (l && o("Access-Control-Allow-Origin", l), s.origin !== "*") {
      const h = a.req.header("Vary");
      h ? o("Vary", h) : o("Vary", "Origin");
    }
    if (s.credentials && o("Access-Control-Allow-Credentials", "true"), (u = s.exposeHeaders) != null && u.length && o("Access-Control-Expose-Headers", s.exposeHeaders.join(",")), a.req.method === "OPTIONS") {
      s.maxAge != null && o("Access-Control-Max-Age", s.maxAge.toString());
      const h = await n(a.req.header("origin") || "", a);
      h.length && o("Access-Control-Allow-Methods", h.join(","));
      let m = s.allowHeaders;
      if (!(m != null && m.length)) {
        const v = a.req.header("Access-Control-Request-Headers");
        v && (m = v.split(/\s*,\s*/));
      }
      return m != null && m.length && (o("Access-Control-Allow-Headers", m.join(",")), a.res.headers.append("Vary", "Access-Control-Request-Headers")), a.res.headers.delete("Content-Length"), a.res.headers.delete("Content-Type"), new Response(null, { headers: a.res.headers, status: 204, statusText: "No Content" });
    }
    await d();
  };
}, "ls");
var ds = /^\s*(?:text\/(?!event-stream(?:[;\s]|$))[^;\s]+|application\/(?:javascript|json|xml|xml-dtd|ecmascript|dart|postscript|rtf|tar|toml|vnd\.dart|vnd\.ms-fontobject|vnd\.ms-opentype|wasm|x-httpd-php|x-javascript|x-ns-proxy-autoconfig|x-sh|x-tar|x-virtualbox-hdd|x-virtualbox-ova|x-virtualbox-ovf|x-virtualbox-vbox|x-virtualbox-vdi|x-virtualbox-vhd|x-virtualbox-vmdk|x-www-form-urlencoded)|font\/(?:otf|ttf)|image\/(?:bmp|vnd\.adobe\.photoshop|vnd\.microsoft\.icon|vnd\.ms-dds|x-icon|x-ms-bmp)|message\/rfc822|model\/gltf-binary|x-shader\/x-fragment|x-shader\/x-vertex|[^;\s]+?\+(?:json|text|xml|yaml))(?:[;\s]|$)/i;
var Ke = /* @__PURE__ */ __name((t, e = hs) => {
  const s = /\.([a-zA-Z0-9]+?)$/, r = t.match(s);
  if (!r) return;
  let n = e[r[1]];
  return n && n.startsWith("text") && (n += "; charset=utf-8"), n;
}, "Ke");
var us = { aac: "audio/aac", avi: "video/x-msvideo", avif: "image/avif", av1: "video/av1", bin: "application/octet-stream", bmp: "image/bmp", css: "text/css", csv: "text/csv", eot: "application/vnd.ms-fontobject", epub: "application/epub+zip", gif: "image/gif", gz: "application/gzip", htm: "text/html", html: "text/html", ico: "image/x-icon", ics: "text/calendar", jpeg: "image/jpeg", jpg: "image/jpeg", js: "text/javascript", json: "application/json", jsonld: "application/ld+json", map: "application/json", mid: "audio/x-midi", midi: "audio/x-midi", mjs: "text/javascript", mp3: "audio/mpeg", mp4: "video/mp4", mpeg: "video/mpeg", oga: "audio/ogg", ogv: "video/ogg", ogx: "application/ogg", opus: "audio/opus", otf: "font/otf", pdf: "application/pdf", png: "image/png", rtf: "application/rtf", svg: "image/svg+xml", tif: "image/tiff", tiff: "image/tiff", ts: "video/mp2t", ttf: "font/ttf", txt: "text/plain", wasm: "application/wasm", webm: "video/webm", weba: "audio/webm", webmanifest: "application/manifest+json", webp: "image/webp", woff: "font/woff", woff2: "font/woff2", xhtml: "application/xhtml+xml", xml: "application/xml", zip: "application/zip", "3gp": "video/3gpp", "3g2": "video/3gpp2", gltf: "model/gltf+json", glb: "model/gltf-binary" };
var hs = us;
var fs = /* @__PURE__ */ __name((...t) => {
  let e = t.filter((n) => n !== "").join("/");
  e = e.replace(new RegExp("(?<=\\/)\\/+", "g"), "");
  const s = e.split("/"), r = [];
  for (const n of s) n === ".." && r.length > 0 && r.at(-1) !== ".." ? r.pop() : n !== "." && r.push(n);
  return r.join("/") || ".";
}, "fs");
var bt = { br: ".br", zstd: ".zst", gzip: ".gz" };
var ps = Object.keys(bt);
var ms = "index.html";
var gs = /* @__PURE__ */ __name((t) => {
  const e = t.root ?? "./", s = t.path, r = t.join ?? fs;
  return async (n, i) => {
    var u, h, m, v;
    if (n.finalized) return i();
    let a;
    if (t.path) a = t.path;
    else try {
      if (a = decodeURIComponent(n.req.path), /(?:^|[\/\\])\.\.(?:$|[\/\\])/.test(a)) throw new Error();
    } catch {
      return await ((u = t.onNotFound) == null ? void 0 : u.call(t, n.req.path, n)), i();
    }
    let d = r(e, !s && t.rewriteRequestPath ? t.rewriteRequestPath(a) : a);
    t.isDir && await t.isDir(d) && (d = r(d, ms));
    const o = t.getContent;
    let l = await o(d, n);
    if (l instanceof Response) return n.newResponse(l.body, l);
    if (l) {
      const O = t.mimes && Ke(d, t.mimes) || Ke(d);
      if (n.header("Content-Type", O || "application/octet-stream"), t.precompressed && (!O || ds.test(O))) {
        const x = new Set((h = n.req.header("Accept-Encoding")) == null ? void 0 : h.split(",").map((w) => w.trim()));
        for (const w of ps) {
          if (!x.has(w)) continue;
          const S = await o(d + bt[w], n);
          if (S) {
            l = S, n.header("Content-Encoding", w), n.header("Vary", "Accept-Encoding", { append: true });
            break;
          }
        }
      }
      return await ((m = t.onFound) == null ? void 0 : m.call(t, d, n)), n.body(l);
    }
    await ((v = t.onNotFound) == null ? void 0 : v.call(t, d, n)), await i();
  };
}, "gs");
var _s = /* @__PURE__ */ __name(async (t, e) => {
  let s;
  e && e.manifest ? typeof e.manifest == "string" ? s = JSON.parse(e.manifest) : s = e.manifest : typeof __STATIC_CONTENT_MANIFEST == "string" ? s = JSON.parse(__STATIC_CONTENT_MANIFEST) : s = __STATIC_CONTENT_MANIFEST;
  let r;
  e && e.namespace ? r = e.namespace : r = __STATIC_CONTENT;
  const n = s[t] || t;
  if (!n) return null;
  const i = await r.get(n, { type: "stream" });
  return i || null;
}, "_s");
var ys = /* @__PURE__ */ __name((t) => async function(s, r) {
  return gs({ ...t, getContent: /* @__PURE__ */ __name(async (i) => _s(i, { manifest: t.manifest, namespace: t.namespace ? t.namespace : s.env ? s.env.__STATIC_CONTENT : void 0 }), "getContent") })(s, r);
}, "ys");
var xs = /* @__PURE__ */ __name((t) => ys(t), "xs");
function P(t, e, s) {
  const r = { x: t.x - e.x, y: t.y - e.y, z: t.z - e.z }, n = { x: s.x - e.x, y: s.y - e.y, z: s.z - e.z }, i = r.x * n.x + r.y * n.y + r.z * n.z, a = Math.sqrt(r.x * r.x + r.y * r.y + r.z * r.z), d = Math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z), l = Math.acos(Math.max(-1, Math.min(1, i / (a * d)))) * (180 / Math.PI);
  return Math.round(l * 10) / 10;
}
__name(P, "P");
function vs(t) {
  const { landmarks: e } = t, s = {};
  try {
    const r = P(e.left_hip, e.left_shoulder, e.left_elbow);
    s.left_shoulder_flexion = { joint_name: "Left Shoulder Flexion", left_angle: r, normal_range: [0, 180], status: r >= 150 ? "normal" : "limited" };
  } catch (r) {
    console.error("Error calculating left shoulder angle:", r);
  }
  try {
    const r = P(e.right_hip, e.right_shoulder, e.right_elbow);
    s.right_shoulder_flexion = { joint_name: "Right Shoulder Flexion", right_angle: r, normal_range: [0, 180], status: r >= 150 ? "normal" : "limited" };
  } catch (r) {
    console.error("Error calculating right shoulder angle:", r);
  }
  try {
    const r = P(e.left_shoulder, e.left_elbow, e.left_wrist);
    s.left_elbow_flexion = { joint_name: "Left Elbow Flexion", left_angle: r, normal_range: [0, 150], status: r >= 130 ? "normal" : "limited" };
  } catch (r) {
    console.error("Error calculating left elbow angle:", r);
  }
  try {
    const r = P(e.right_shoulder, e.right_elbow, e.right_wrist);
    s.right_elbow_flexion = { joint_name: "Right Elbow Flexion", right_angle: r, normal_range: [0, 150], status: r >= 130 ? "normal" : "limited" };
  } catch (r) {
    console.error("Error calculating right elbow angle:", r);
  }
  try {
    const r = P(e.left_shoulder, e.left_hip, e.left_knee);
    s.left_hip_flexion = { joint_name: "Left Hip Flexion", left_angle: r, normal_range: [0, 120], status: r >= 90 ? "normal" : "limited" };
  } catch (r) {
    console.error("Error calculating left hip angle:", r);
  }
  try {
    const r = P(e.right_shoulder, e.right_hip, e.right_knee);
    s.right_hip_flexion = { joint_name: "Right Hip Flexion", right_angle: r, normal_range: [0, 120], status: r >= 90 ? "normal" : "limited" };
  } catch (r) {
    console.error("Error calculating right hip angle:", r);
  }
  try {
    const r = P(e.left_hip, e.left_knee, e.left_ankle);
    s.left_knee_flexion = { joint_name: "Left Knee Flexion", left_angle: r, normal_range: [0, 135], status: r >= 120 ? "normal" : "limited" };
  } catch (r) {
    console.error("Error calculating left knee angle:", r);
  }
  try {
    const r = P(e.right_hip, e.right_knee, e.right_ankle);
    s.right_knee_flexion = { joint_name: "Right Knee Flexion", right_angle: r, normal_range: [0, 135], status: r >= 120 ? "normal" : "limited" };
  } catch (r) {
    console.error("Error calculating right knee angle:", r);
  }
  try {
    const r = P(e.left_knee, e.left_ankle, e.left_foot_index);
    s.left_ankle_dorsiflexion = { joint_name: "Left Ankle Dorsiflexion", left_angle: r, normal_range: [70, 110], status: r >= 85 && r <= 105 ? "normal" : "limited" };
  } catch (r) {
    console.error("Error calculating left ankle angle:", r);
  }
  try {
    const r = P(e.right_knee, e.right_ankle, e.right_foot_index);
    s.right_ankle_dorsiflexion = { joint_name: "Right Ankle Dorsiflexion", right_angle: r, normal_range: [70, 110], status: r >= 85 && r <= 105 ? "normal" : "limited" };
  } catch (r) {
    console.error("Error calculating right ankle angle:", r);
  }
  for (const r in s) {
    const n = s[r];
    n.left_angle !== void 0 && n.right_angle !== void 0 && (n.bilateral_difference = Math.abs(n.left_angle - n.right_angle));
  }
  return s;
}
__name(vs, "vs");
var ws = [["left_shoulder_flexion", "right_shoulder_flexion", "shoulder"], ["left_elbow_flexion", "right_elbow_flexion", "elbow"], ["left_hip_flexion", "right_hip_flexion", "hip"], ["left_knee_flexion", "right_knee_flexion", "knee"], ["left_ankle_dorsiflexion", "right_ankle_dorsiflexion", "ankle"]];
function Es(t) {
  const e = {};
  for (const [s, r, n] of ws) {
    const i = t[s], a = t[r];
    if ((i == null ? void 0 : i.left_angle) !== void 0 && (a == null ? void 0 : a.right_angle) !== void 0) {
      const o = Math.abs(i.left_angle - a.right_angle) / Math.max(i.left_angle, a.right_angle) * 100;
      o > 10 && (e[n] = Math.round(o * 10) / 10);
    }
  }
  return e;
}
__name(Es, "Es");
function bs(t, e) {
  const s = [], { landmarks: r } = t;
  try {
    const n = r.left_knee.x, i = r.right_knee.x, a = r.left_ankle.x, d = r.right_ankle.x;
    Math.abs(n - a) < 0.05 && s.push("Left knee valgus detected - knee tracking inward"), Math.abs(i - d) < 0.05 && s.push("Right knee valgus detected - knee tracking inward");
    const o = { x: r.left_shoulder.x - r.left_hip.x, y: r.left_shoulder.y - r.left_hip.y };
    Math.atan2(o.x, Math.abs(o.y)) * (180 / Math.PI) > 30 && s.push("Excessive forward trunk lean - core weakness or hip mobility limitation");
    const u = Math.abs(r.left_heel.y - r.left_foot_index.y), h = Math.abs(r.right_heel.y - r.right_foot_index.y);
    u > 0.15 && s.push("Left heel lifting - ankle dorsiflexion limitation"), h > 0.15 && s.push("Right heel lifting - ankle dorsiflexion limitation"), Math.abs(r.left_shoulder.y - r.right_shoulder.y) > 0.08 && s.push("Shoulder height asymmetry - possible lateral trunk lean or shoulder dysfunction"), Math.abs(r.left_hip.y - r.right_hip.y) > 0.08 && s.push("Pelvic obliquity - one hip higher than the other");
  } catch (n) {
    console.error("Error detecting compensations:", n);
  }
  return s;
}
__name(bs, "bs");
function Rs(t, e) {
  let s = 100;
  for (const r in t) {
    const n = t[r];
    n.status === "limited" ? s -= 10 : n.status === "excessive" && (s -= 5);
  }
  for (const r in t) {
    const n = t[r];
    n.bilateral_difference && n.bilateral_difference > 10 && (s -= 8);
  }
  return s -= e.length * 7, Math.max(0, Math.min(100, s));
}
__name(Rs, "Rs");
function js(t, e, s) {
  const r = [], n = t.left_ankle_dorsiflexion, i = t.right_ankle_dorsiflexion;
  ((n == null ? void 0 : n.status) === "limited" || (i == null ? void 0 : i.status) === "limited") && r.push({ area: "Ankle Dorsiflexion", severity: "moderate", description: "Limited ankle dorsiflexion range of motion. This can lead to heel lifting during squats and compensation patterns up the kinetic chain.", recommended_exercises: [3] });
  const a = t.left_hip_flexion, d = t.right_hip_flexion;
  ((a == null ? void 0 : a.status) === "limited" || (d == null ? void 0 : d.status) === "limited") && r.push({ area: "Hip Flexion", severity: "moderate", description: "Limited hip flexion range of motion. This affects squat depth and can cause excessive forward trunk lean.", recommended_exercises: [1, 3] });
  const o = t.left_shoulder_flexion, l = t.right_shoulder_flexion;
  if (((o == null ? void 0 : o.status) === "limited" || (l == null ? void 0 : l.status) === "limited") && r.push({ area: "Shoulder Flexion", severity: "mild", description: "Limited overhead shoulder mobility. This can affect overhead movements and daily activities.", recommended_exercises: [11, 12] }), e.some((u) => u.includes("knee valgus") || u.includes("balance")) && r.push({ area: "Lower Extremity Stability", severity: "moderate", description: "Instability and poor movement control in lower extremity. Knee valgus and balance deficits detected.", recommended_exercises: [4, 5] }), e.some((u) => u.includes("trunk lean") || u.includes("core")) && r.push({ area: "Core Stability", severity: "moderate", description: "Weak core stability causing compensatory trunk movements and poor postural control.", recommended_exercises: [5, 13, 14] }), Object.keys(s).length > 0) {
    const u = Math.max(...Object.values(s));
    r.push({ area: "Bilateral Asymmetry", severity: u > 20 ? "severe" : u > 15 ? "moderate" : "mild", description: `Significant differences between left and right sides detected. Asymmetries found in: ${Object.keys(s).join(", ")}. This increases injury risk and affects functional performance.`, recommended_exercises: [7, 8] });
  }
  return r;
}
__name(js, "js");
function Ss(t) {
  const e = vs(t), s = Es(e), r = bs(t), n = Rs(e, r), i = js(e, r, s), a = Ts(e, r, i);
  return { joint_angles: Object.values(e), movement_quality_score: n, detected_compensations: r, recommendations: a, deficiencies: i };
}
__name(Ss, "Ss");
function Ts(t, e, s) {
  const r = [];
  return r.push("Continue with prescribed exercise program focusing on identified deficiencies"), e.length > 3 && r.push("Multiple compensation patterns detected - recommend reducing exercise complexity until fundamental movement patterns improve"), s.some((n) => n.severity === "severe") && r.push("Severe deficiencies identified - recommend twice-weekly supervised therapy sessions"), Object.keys(t).some((n) => t[n].status === "limited") && r.push("Perform daily mobility work for 10-15 minutes focusing on identified ROM limitations"), r.push("Re-assess movement quality in 2-3 weeks to track progress"), r;
}
__name(Ts, "Ts");
async function Rt(t, e) {
  const s = e.toLowerCase().split(" ").filter((d) => d.length > 3);
  if (s.length === 0) return { answer: "I'm sorry, I couldn't find enough specific information in your query to provide clinical guidance.", sources: [], confidence: 0 };
  let r = [];
  try {
    const { results: d } = await t.prepare(`
      SELECT * FROM exercises
    `).all();
    r = d;
  } catch (d) {
    console.error("RAG Database error:", d);
  }
  const n = r.filter((d) => {
    const o = `${d.name || ""} ${d.description || ""} ${d.instructions || ""}`.toLowerCase();
    return s.some((l) => o.includes(l));
  });
  if (n.length === 0) return { answer: "Based on our clinical library, I don't have specific training protocols for that movement yet. I recommend focusing on fundamental mobility exercises.", sources: [], confidence: 0.1 };
  const i = n[0];
  return { answer: `Based on our clinical protocols, for concerns related to "${e}", we recommend focusing on ${i.name}. ${i.description}. To perform this correctly: ${(i.instructions || "").split(`
`)[0]}.`, sources: n.slice(0, 3).map((d) => d.name), confidence: 0.85 };
}
__name(Rt, "Rt");
var _ = new Et();
_.use("/api/*", ls());
_.use("/static/*", xs({ root: "./public", manifest: {} }));
var Pe = null;
var Ve = 0;
var Os = 3600 * 1e3;
var De = null;
var Je = 0;
var As = 3600 * 1e3;
_.post("/api/auth/register", async (t) => {
  try {
    const e = await t.req.json();
    if (await t.env.DB.prepare(`
      SELECT id FROM clinicians WHERE email = ?
    `).bind(e.email).first()) return t.json({ success: false, error: "Email already registered" }, 400);
    const r = await jt(e.password), n = await t.env.DB.prepare(`
      INSERT INTO clinicians (
        email, password_hash, first_name, last_name, title,
        license_number, license_state, npi_number, phone, clinic_name
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(e.email, r, e.first_name, e.last_name, e.title, e.license_number, e.license_state, e.npi_number, e.phone, e.clinic_name).run();
    return t.json({ success: true, data: { id: n.meta.last_row_id } });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.post("/api/auth/login", async (t) => {
  try {
    const { email: e, password: s } = await t.req.json(), r = await t.env.DB.prepare(`
      SELECT * FROM clinicians WHERE email = ? AND active = 1
    `).bind(e).first();
    if (!r) return t.json({ success: false, error: "Invalid email or password" }, 401);
    if (!await Cs(s, r.password_hash)) return t.json({ success: false, error: "Invalid email or password" }, 401);
    await t.env.DB.prepare(`
      UPDATE clinicians SET last_login = CURRENT_TIMESTAMP WHERE id = ?
    `).bind(r.id).run();
    const { password_hash: i, ...a } = r;
    return t.json({ success: true, data: a });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.get("/api/auth/profile/:id", async (t) => {
  try {
    const e = t.req.param("id"), s = await t.env.DB.prepare(`
      SELECT id, email, first_name, last_name, title, license_number,
             license_state, npi_number, phone, clinic_name, role, active,
             created_at, last_login
      FROM clinicians WHERE id = ?
    `).bind(e).first();
    return s ? t.json({ success: true, data: s }) : t.json({ success: false, error: "Clinician not found" }, 404);
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.post("/api/patients", async (t) => {
  try {
    const e = await t.req.json(), s = await t.env.DB.prepare(`
      INSERT INTO patients (
        first_name, last_name, date_of_birth, gender, email, phone,
        emergency_contact_name, emergency_contact_phone,
        address_line1, city, state, zip_code,
        height_cm, weight_kg, insurance_provider
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(e.first_name, e.last_name, e.date_of_birth, e.gender, e.email, e.phone, e.emergency_contact_name, e.emergency_contact_phone, e.address_line1, e.city, e.state, e.zip_code, e.height_cm, e.weight_kg, e.insurance_provider).run();
    return t.json({ success: true, data: { id: s.meta.last_row_id, ...e } });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.get("/api/patients", async (t) => {
  try {
    const { results: e } = await t.env.DB.prepare(`
      SELECT * FROM patients ORDER BY created_at DESC
    `).all();
    return t.json({ success: true, data: e });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.get("/api/patients/:id", async (t) => {
  try {
    const e = t.req.param("id"), s = await t.env.DB.prepare(`
      SELECT * FROM patients WHERE id = ?
    `).bind(e).first();
    return s ? t.json({ success: true, data: s }) : t.json({ success: false, error: "Patient not found" }, 404);
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.post("/api/patients/:id/medical-history", async (t) => {
  try {
    const e = t.req.param("id"), s = await t.req.json(), r = await t.env.DB.prepare(`
      INSERT INTO medical_history (
        patient_id, surgery_type, surgery_date, conditions, medications, allergies,
        current_pain_level, pain_location, activity_level, treatment_goals
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(e, s.surgery_type, s.surgery_date, JSON.stringify(s.conditions), JSON.stringify(s.medications), JSON.stringify(s.allergies), s.current_pain_level, JSON.stringify(s.pain_location), s.activity_level, s.treatment_goals).run();
    return t.json({ success: true, data: { id: r.meta.last_row_id } });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.post("/api/assessments", async (t) => {
  try {
    const e = await t.req.json(), s = await t.env.DB.prepare(`
      INSERT INTO assessments (
        patient_id, clinician_id, assessment_type, status
      ) VALUES (?, ?, ?, ?)
    `).bind(e.patient_id, e.clinician_id || 1, e.assessment_type, "in_progress").run();
    return t.json({ success: true, data: { id: s.meta.last_row_id, ...e } });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.get("/api/patients/:id/assessments", async (t) => {
  try {
    const e = t.req.param("id"), { results: s } = await t.env.DB.prepare(`
      SELECT * FROM assessments WHERE patient_id = ? ORDER BY assessment_date DESC
    `).bind(e).all();
    return t.json({ success: true, data: s });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.get("/api/assessments", async (t) => {
  try {
    const { results: e } = await t.env.DB.prepare(`
      SELECT a.*, p.first_name, p.last_name
      FROM assessments a
      JOIN patients p ON a.patient_id = p.id
      ORDER BY a.assessment_date DESC
    `).all();
    return t.json({ success: true, data: e });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.get("/api/assessments/:id", async (t) => {
  try {
    const e = t.req.param("id"), s = await t.env.DB.prepare(`
      SELECT a.*, p.first_name, p.last_name, p.date_of_birth
      FROM assessments a
      JOIN patients p ON a.patient_id = p.id
      WHERE a.id = ?
    `).bind(e).first();
    return s ? t.json({ success: true, data: s }) : t.json({ success: false, error: "Assessment not found" }, 404);
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.post("/api/assessments/:id/tests", async (t) => {
  try {
    const e = t.req.param("id"), s = await t.req.json(), r = await t.env.DB.prepare(`
      INSERT INTO movement_tests (
        assessment_id, test_name, test_category, test_order, instructions, status
      ) VALUES (?, ?, ?, ?, ?, ?)
    `).bind(e, s.test_name, s.test_category, s.test_order, s.instructions, "pending").run();
    return t.json({ success: true, data: { id: r.meta.last_row_id } });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.post("/api/tests/:id/analyze", async (t) => {
  try {
    const e = t.req.param("id"), { skeleton_data: s } = await t.req.json(), r = Ss(s);
    return await t.env.DB.prepare(`
      UPDATE movement_tests
      SET skeleton_data = ?,
          movement_quality_score = ?,
          deficiencies = ?,
          compensations_detected = ?,
          status = 'completed',
          completed_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `).bind(JSON.stringify(s), r.movement_quality_score, JSON.stringify(r.deficiencies), JSON.stringify(r.detected_compensations), e).run(), t.json({ success: true, data: { movement_quality_score: r.movement_quality_score, deficiencies: r.deficiencies, compensations: r.detected_compensations, recommendations: r.recommendations } });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.get("/api/tests/:id/results", async (t) => {
  try {
    const e = t.req.param("id"), s = await t.env.DB.prepare(`
      SELECT * FROM movement_tests WHERE id = ?
    `).bind(e).first();
    return s ? t.json({ success: true, data: s }) : t.json({ success: false, error: "Test not found" }, 404);
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.get("/api/assessments/:id/tests", async (t) => {
  try {
    const e = t.req.param("id"), { results: s } = await t.env.DB.prepare(`
      SELECT * FROM movement_tests WHERE assessment_id = ? ORDER BY test_order
    `).bind(e).all();
    return t.json({ success: true, data: s });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.get("/api/exercises", async (t) => {
  try {
    const e = t.req.query("category"), s = Date.now();
    if (!e && Pe && s - Ve < Os) return t.json({ success: true, data: Pe });
    let r = "SELECT * FROM exercises";
    const n = [];
    e && (r += " WHERE category = ?", n.push(e)), r += " ORDER BY name";
    const { results: i } = await t.env.DB.prepare(r).bind(...n).all();
    return e || (Pe = i, Ve = s), t.json({ success: true, data: i });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.post("/api/prescriptions", async (t) => {
  try {
    const e = await t.req.json(), s = await t.env.DB.prepare(`
      INSERT INTO prescribed_exercises (
        patient_id, assessment_id, exercise_id, sets, repetitions,
        times_per_week, clinical_reason, target_deficiency, prescribed_by
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(e.patient_id, e.assessment_id, e.exercise_id, e.sets, e.repetitions, e.times_per_week, e.clinical_reason, e.target_deficiency, e.prescribed_by || 1).run();
    return t.json({ success: true, data: { id: s.meta.last_row_id } });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.get("/api/patients/:id/prescriptions", async (t) => {
  try {
    const e = t.req.param("id"), { results: s } = await t.env.DB.prepare(`
      SELECT
        pe.*,
        e.name as exercise_name,
        e.description,
        e.instructions,
        e.demo_video_url,
        e.difficulty
      FROM prescribed_exercises pe
      JOIN exercises e ON pe.exercise_id = e.id
      WHERE pe.patient_id = ? AND pe.status = 'active'
      ORDER BY pe.created_at DESC
    `).bind(e).all();
    return t.json({ success: true, data: s });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.post("/api/exercise-sessions", async (t) => {
  try {
    const e = await t.req.json(), s = await t.env.DB.prepare(`
      INSERT INTO exercise_sessions (
        patient_id, prescribed_exercise_id, sets_completed, reps_completed,
        duration_seconds, form_quality_score, pose_accuracy_data,
        pain_level_during, difficulty_rating, completed
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(e.patient_id, e.prescribed_exercise_id, e.sets_completed, e.reps_completed, e.duration_seconds, e.form_quality_score, e.pose_accuracy_data, e.pain_level_during, e.difficulty_rating, e.completed).run();
    return await Ms(t.env.DB, e.prescribed_exercise_id), t.json({ success: true, data: { id: s.meta.last_row_id } });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.get("/api/patients/:id/sessions", async (t) => {
  try {
    const e = t.req.param("id"), { results: s } = await t.env.DB.prepare(`
      SELECT
        es.*,
        e.name as exercise_name,
        pe.sets as prescribed_sets,
        pe.repetitions as prescribed_reps
      FROM exercise_sessions es
      JOIN prescribed_exercises pe ON es.prescribed_exercise_id = pe.id
      JOIN exercises e ON pe.exercise_id = e.id
      WHERE es.patient_id = ?
      ORDER BY es.session_date DESC
      LIMIT 50
    `).bind(e).all();
    return t.json({ success: true, data: s });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.get("/api/billing/codes", async (t) => {
  try {
    const e = Date.now();
    if (De && e - Je < As) return t.json({ success: true, data: De });
    const { results: s } = await t.env.DB.prepare(`
      SELECT * FROM billing_codes ORDER BY cpt_code
    `).all();
    return De = s, Je = e, t.json({ success: true, data: s });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.post("/api/billing/events", async (t) => {
  try {
    const e = await t.req.json(), s = await t.env.DB.prepare(`
      INSERT INTO billable_events (
        patient_id, assessment_id, exercise_session_id,
        cpt_code_id, service_date, duration_minutes,
        clinical_note, provider_id
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(e.patient_id, e.assessment_id, e.exercise_session_id, e.cpt_code_id, e.service_date, e.duration_minutes, e.clinical_note, e.provider_id || 1).run();
    return t.json({ success: true, data: { id: s.meta.last_row_id } });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.post("/api/rag/query", async (t) => {
  try {
    const { query: e } = await t.req.json(), s = await Rt(t.env.DB, e);
    return t.json({ success: true, data: s });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
_.post("/api/assessments/:id/generate-note", async (t) => {
  try {
    const e = t.req.param("id"), s = await t.env.DB.prepare(`
      SELECT * FROM assessments WHERE id = ?
    `).bind(e).first(), { results: r } = await t.env.DB.prepare(`
      SELECT mt.*, ma.*
      FROM movement_tests mt
      LEFT JOIN movement_analysis ma ON mt.id = ma.test_id
      WHERE mt.assessment_id = ?
    `).bind(e).all(), n = Ns(s, r);
    let i = "";
    if (r.length > 0 && r[0].deficiencies) try {
      const a = JSON.parse(r[0].deficiencies);
      a.length > 0 && (i = (await Rt(t.env.DB, a[0].area)).answer);
    } catch {
    }
    return i && (n.plan += `

AI CLINICAL INSIGHTS:
${i}`), await t.env.DB.prepare(`
      UPDATE assessments
      SET subjective_findings = ?,
          objective_findings = ?,
          assessment_summary = ?,
          plan = ?,
          status = 'completed'
      WHERE id = ?
    `).bind(n.subjective, n.objective, n.assessment, n.plan, e).run(), t.json({ success: true, data: n });
  } catch (e) {
    return t.json({ success: false, error: e.message }, 500);
  }
});
async function jt(t) {
  const s = new TextEncoder().encode(t + "physiomotion-salt-2025"), r = await crypto.subtle.digest("SHA-256", s);
  return Array.from(new Uint8Array(r)).map((a) => a.toString(16).padStart(2, "0")).join("");
}
__name(jt, "jt");
async function Cs(t, e) {
  return await jt(t) === e;
}
__name(Cs, "Cs");
async function Ms(t, e) {
  const s = await t.prepare(`
    SELECT COUNT(*) as completed_count
    FROM exercise_sessions
    WHERE prescribed_exercise_id = ? AND completed = 1
  `).bind(e).first(), r = await t.prepare(`
    SELECT times_per_week, prescribed_at FROM prescribed_exercises WHERE id = ?
  `).bind(e).first();
  if (s && r) {
    const n = new Date(r.prescribed_at), i = /* @__PURE__ */ new Date();
    if (n > i) return;
    const a = Math.max(1, Math.floor((i.getTime() - n.getTime()) / (10080 * 60 * 1e3))), d = r.times_per_week * a, o = Math.min(100, Math.round(s.completed_count / d * 100));
    await t.prepare(`
      UPDATE prescribed_exercises
      SET compliance_percentage = ?,
          last_performed_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `).bind(o, e).run();
  }
}
__name(Ms, "Ms");
function Ns(t, e) {
  const s = [], r = [];
  let n = 0;
  for (const l of e) {
    if (l.deficiencies) try {
      const u = JSON.parse(l.deficiencies);
      s.push(...u);
    } catch {
    }
    if (l.ai_recommendations) try {
      const u = JSON.parse(l.ai_recommendations);
      r.push(...u);
    } catch {
    }
    l.movement_quality_score && (n += l.movement_quality_score);
  }
  n = e.length > 0 ? Math.round(n / e.length) : 0;
  const i = `Patient presents for ${t.assessment_type} assessment. Willing and able to participate in functional movement screening.`, a = `
FUNCTIONAL MOVEMENT ASSESSMENT RESULTS:

Overall Movement Quality Score: ${n}/100

Tests Completed: ${e.length}
${e.map((l) => `- ${l.test_name}: ${l.test_status}`).join(`
`)}

DEFICIENCIES IDENTIFIED:
${s.map((l, u) => `${u + 1}. ${l.area} (${l.severity}): ${l.description}`).join(`

`)}

Movement Quality: ${n >= 80 ? "Good" : n >= 60 ? "Fair" : "Poor"}
Compensatory patterns observed and documented in biomechanical analysis.
  `.trim(), d = `
Patient demonstrates ${n >= 80 ? "good" : n >= 60 ? "fair" : "poor"} movement quality with ${s.length} significant deficiencies identified.

PRIMARY FINDINGS:
${s.slice(0, 3).map((l, u) => `${u + 1}. ${l.area} - ${l.severity} severity`).join(`
`)}

FUNCTIONAL IMPACT:
${n < 60 ? "Significant functional limitations present. Patient would benefit from comprehensive therapeutic exercise program." : n < 80 ? "Moderate functional limitations. Targeted interventions recommended." : "Minor limitations identified. Preventive exercise program appropriate."}
  `.trim(), o = `
TREATMENT PLAN:

1. THERAPEUTIC EXERCISES: Prescribed evidence-based exercise program targeting identified deficiencies
   ${s.slice(0, 3).map((l) => `   - Address ${l.area}`).join(`
`)}

2. FREQUENCY: ${n < 60 ? "2-3x per week supervised therapy + daily HEP" : "1-2x per week supervised + daily HEP"}

3. DURATION: ${n < 60 ? "8-12 weeks" : "4-8 weeks"}

4. REMOTE MONITORING: Patient enrolled in remote therapeutic monitoring program (RTM)
   - Daily exercise compliance tracking via mobile app
   - Real-time form analysis and feedback
   - Weekly progress reports

5. RE-ASSESSMENT: Schedule follow-up functional movement assessment in 4 weeks

6. PATIENT EDUCATION:
${r.slice(0, 3).map((l) => `   - ${l}`).join(`
`)}

CPT CODES: 97163, 97110, 97112, 98975, 98977
  `.trim();
  return { subjective: i, objective: a, assessment: d, plan: o };
}
__name(Ns, "Ns");
_.get("/assessment", (t) => t.redirect("/static/assessment"));
_.get("/intake", (t) => t.redirect("/static/intake"));
_.get("/patients", (t) => t.redirect("/static/patients"));
_.get("/login", (t) => t.redirect("/static/login.html"));
_.get("/register", (t) => t.redirect("/static/register.html"));
_.get("/", (t) => t.html(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PhysioMotion - Medical Movement Assessment Platform</title>
        <script src="https://cdn.tailwindcss.com"><\/script>
        <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet">
    </head>
    <body class="bg-gray-50">
        <script>
            // Check authentication on page load
            const session = localStorage.getItem('clinician_session') || sessionStorage.getItem('clinician_session');
            if (!session) {
                window.location.href = '/static/login.html';
            }
        <\/script>
        <div id="app">
            <!-- Navigation -->
            <nav class="bg-white shadow-lg border-b-2 border-cyan-500">
                <div class="max-w-7xl mx-auto px-4">
                    <div class="flex justify-between h-16">
                        <div class="flex items-center">
                            <i class="fas fa-heartbeat text-cyan-600 text-2xl mr-3"></i>
                            <span class="text-xl font-bold text-slate-800">PhysioMotion</span>
                        </div>
                        <div class="flex items-center space-x-4">
                            <a href="/" class="text-gray-700 hover:text-cyan-600 transition-colors"><i class="fas fa-home mr-2"></i>Home</a>
                            <a href="/patients" class="text-gray-700 hover:text-cyan-600 transition-colors"><i class="fas fa-users mr-2"></i>Patients</a>
                            <a href="/intake" class="text-gray-700 hover:text-cyan-600 transition-colors"><i class="fas fa-user-plus mr-2"></i>New Patient</a>
                            <a href="/assessment" class="text-gray-700 hover:text-cyan-600 transition-colors"><i class="fas fa-video mr-2"></i>Assessment</a>
                            <div class="flex items-center space-x-3 ml-4 pl-4 border-l border-gray-300">
                                <span id="clinicianName" class="text-gray-700 font-medium"></span>
                                <button onclick="logout()" class="text-red-600 hover:text-red-700 transition-colors">
                                    <i class="fas fa-sign-out-alt mr-1"></i>Logout
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </nav>

            <!-- Hero Section -->
            <div class="bg-gradient-to-r from-cyan-600 to-purple-600 text-white py-16">
                <div class="max-w-7xl mx-auto px-4 text-center">
                    <h1 class="text-4xl font-bold mb-4">Medical Movement Assessment Platform</h1>
                    <p class="text-xl mb-8">AI-Powered Biomechanical Analysis for Physical Therapy & Chiropractic Care</p>
                    <div class="flex justify-center space-x-4">
                        <button onclick="window.location.href='/intake'" class="bg-white text-cyan-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-all">
                            <i class="fas fa-user-plus mr-2"></i>New Patient Intake
                        </button>
                        <button onclick="window.location.href='/assessment'" class="bg-purple-500 text-white px-6 py-3 rounded-lg font-semibold hover:bg-purple-400 transition-all">
                            <i class="fas fa-video mr-2"></i>Start Assessment
                        </button>
                    </div>
                </div>
            </div>

            <!-- Features Grid -->
            <div class="max-w-7xl mx-auto px-4 py-12">
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <!-- Feature 1 -->
                    <div class="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
                        <div class="text-cyan-600 text-3xl mb-4"><i class="fas fa-camera"></i></div>
                        <h3 class="text-xl font-bold mb-2">Real-Time Motion Capture</h3>
                        <p class="text-gray-600">Advanced Orbbec Femto Mega integration with Azure Kinect Body Tracking SDK for professional clinical assessments</p>
                    </div>

                    <!-- Feature 2 -->
                    <div class="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
                        <div class="text-purple-600 text-3xl mb-4"><i class="fas fa-brain"></i></div>
                        <h3 class="text-xl font-bold mb-2">AI Biomechanical Analysis</h3>
                        <p class="text-gray-600">Automated joint angle calculations, ROM measurements, and compensation pattern detection</p>
                    </div>

                    <!-- Feature 3 -->
                    <div class="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
                        <div class="text-violet-600 text-3xl mb-4"><i class="fas fa-mobile-alt"></i></div>
                        <h3 class="text-xl font-bold mb-2">Home Exercise Monitoring</h3>
                        <p class="text-gray-600">MediaPipe Pose integration for remote patient monitoring via mobile camera with real-time feedback</p>
                    </div>

                    <!-- Feature 4 -->
                    <div class="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
                        <div class="text-rose-600 text-3xl mb-4"><i class="fas fa-file-medical"></i></div>
                        <h3 class="text-xl font-bold mb-2">Automated Medical Notes</h3>
                        <p class="text-gray-600">AI-generated SOAP notes with comprehensive deficiency documentation and treatment plans</p>
                    </div>

                    <!-- Feature 5 -->
                    <div class="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
                        <div class="text-teal-600 text-3xl mb-4"><i class="fas fa-dumbbell"></i></div>
                        <h3 class="text-xl font-bold mb-2">Exercise Prescription</h3>
                        <p class="text-gray-600">Evidence-based exercise library with automated prescription based on identified deficiencies</p>
                    </div>

                    <!-- Feature 6 -->
                    <div class="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
                        <div class="text-indigo-600 text-3xl mb-4"><i class="fas fa-dollar-sign"></i></div>
                        <h3 class="text-xl font-bold mb-2">Medical Billing</h3>
                        <p class="text-gray-600">Integrated CPT coding with RPM/RTM billing support for remote patient monitoring</p>
                    </div>
                </div>
            </div>

            <!-- Stats Section -->
            <div class="bg-gradient-to-r from-slate-50 to-slate-100 py-12">
                <div class="max-w-7xl mx-auto px-4">
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-6 text-center">
                        <div>
                            <div class="text-4xl font-bold text-cyan-600">32</div>
                            <div class="text-gray-600">Joint Points Tracked</div>
                        </div>
                        <div>
                            <div class="text-4xl font-bold text-purple-600">15+</div>
                            <div class="text-gray-600">Evidence-Based Exercises</div>
                        </div>
                        <div>
                            <div class="text-4xl font-bold text-violet-600">Real-Time</div>
                            <div class="text-gray-600">Analysis & Feedback</div>
                        </div>
                        <div>
                            <div class="text-4xl font-bold text-teal-600">92%</div>
                            <div class="text-gray-600">AI Confidence Score</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Footer -->
            <footer class="bg-gradient-to-r from-slate-800 to-slate-900 text-white py-8 border-t-4 border-cyan-500">
                <div class="max-w-7xl mx-auto px-4 text-center">
                    <p>&copy; 2025 PhysioMotion. Medical-Grade Movement Assessment Platform.</p>
                    <p class="text-sm text-slate-400 mt-2">Powered by Orbbec Femto Mega, Azure Kinect Body Tracking SDK, and MediaPipe Pose</p>
                </div>
            </footer>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/axios@1.6.0/dist/axios.min.js"><\/script>
        <script>
            // Display logged-in clinician info
            function displayClinicianInfo() {
                const session = localStorage.getItem('clinician_session') || sessionStorage.getItem('clinician_session');
                if (session) {
                    try {
                        const clinician = JSON.parse(session);
                        const nameElement = document.getElementById('clinicianName');
                        if (nameElement) {
                            const demoLabel = clinician.is_demo ? ' <span class="text-xs bg-violet-500 text-white px-2 py-1 rounded">DEMO</span>' : '';
                            nameElement.innerHTML = '<i class="fas fa-user-md mr-1"></i>' + clinician.first_name + ' ' + clinician.last_name + (clinician.title ? ', ' + clinician.title : '') + demoLabel;
                        }
                    } catch (e) {
                        console.error('Error parsing session:', e);
                    }
                }
            }

            // Logout function
            function logout() {
                if (confirm('Are you sure you want to logout?')) {
                    localStorage.removeItem('clinician_session');
                    sessionStorage.removeItem('clinician_session');
                    window.location.href = '/static/login.html';
                }
            }

            // Initialize on page load
            displayClinicianInfo();
        <\/script>
    </body>
    </html>
  `));
var Le = new Et();
var St = Object.assign({ "/src/index.tsx": _ });
var Tt = false;
for (const [, t] of Object.entries(St)) t && (Le.all("*", (e) => {
  let s;
  try {
    s = e.executionCtx;
  } catch {
  }
  return t.fetch(e.req.raw, e.env, s);
}), Le.notFound((e) => {
  let s;
  try {
    s = e.executionCtx;
  } catch {
  }
  return t.fetch(e.req.raw, e.env, s);
}), Tt = true);
if (!Tt) throw new Error("Can't import modules from ['/src/index.ts','/src/index.tsx','/app/server.ts']");
var Ot = {};
var Ge = /* @__PURE__ */ new Set();
for (const [t, e] of Object.entries(St)) for (const [s, r] of Object.entries(e)) if (s !== "fetch") {
  if (Ge.has(s)) throw new Error(`Handler "${s}" is defined in multiple entry files. Please ensure each handler (except fetch) is defined only once.`);
  Ge.add(s), Ot[s] = r;
}
var Hs = { ...Ot, fetch: Le.fetch };
export {
  Hs as default
};
//# sourceMappingURL=index.js.map
