import {
  DrawingUtils,
  FilesetResolver,
  GestureRecognizer,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/+esm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/gesture_recognizer.task";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

const HOLD_DURATION_MS = 650;
const CLOSED_FIST_THRESHOLD = 0.68;
const FLICKER_GRACE_MS = 120;
const INFERENCE_INTERVAL_MS = 38;
const LOADING_DURATION_MS = 4600;
const POINTER_CLICK_DURATION_MS = 520;
const POINTER_SMOOTHING = 0.46;
const SWIPE_STILL_MS = 150;
const SWIPE_MAX_MS = 480;
const SWIPE_LEFT_DISTANCE = 0.16;
const SWIPE_COOLDOWN_MS = 1000;
const POINTER_CAMERA_RANGE = {
  loading: { x: 0.62, y: 0.60 },
  selection: { x: 0.62, y: 0.62 },
  tiger: { x: 0.52, y: 0.52 },
};

const ui = {
  experience: document.querySelector(".experience"),
  startScreen: document.querySelector("#startScreen"),
  loadingScreen: document.querySelector("#loadingScreen"),
  selectionScreen: document.querySelector("#selectionScreen"),
  selectionBadges: [...document.querySelectorAll(".selection-badge")],
  selectionHint: document.querySelector("#selectionHint"),
  tigerScreen: document.querySelector("#tigerScreen"),
  tigerArt: document.querySelector("#tigerArt"),
  tigerArtIncoming: document.querySelector("#tigerArtIncoming"),
  tigerBack: document.querySelector("#tigerBack"),
  tigerNext: document.querySelector("#tigerNext"),
  tigerSteps: [...document.querySelectorAll(".tiger-step")],
  tigerTargets: [...document.querySelectorAll(".tiger-hit")],
  capturePanel: document.querySelector("#capturePanel"),
  video: document.querySelector("#cameraVideo"),
  canvas: document.querySelector("#landmarkCanvas"),
  gestureLabel: document.querySelector("#gestureLabel"),
  statusLabel: document.querySelector("#statusLabel"),
  fpsLabel: document.querySelector("#fpsLabel"),
  progressRing: document.querySelector("#progressRing"),
  progressValue: document.querySelector("#progressValue"),
  retryButton: document.querySelector("#retryButton"),
  loadingMotion: document.querySelector(".loading-motion"),
  loadingTitleCanvas: document.querySelector("#loadingTitleCanvas"),
  loadingFill: document.querySelector("#loadingFill"),
  loadingPhase: document.querySelector("#loadingPhase"),
  loadingPercent: document.querySelector("#loadingPercent"),
  backButton: document.querySelector("#backButton"),
  gesturePointer: document.querySelector("#gesturePointer"),
  pointerStatus: document.querySelector("#pointerStatus"),
};

let recognizer = null;
let stream = null;
let drawingUtils = null;
let lastVideoTime = -1;
let lastInferenceAt = 0;
let fistStartedAt = null;
let lastClosedFistAt = -Infinity;
let running = false;
let triggered = false;
let frameCounter = 0;
let fpsWindowStartedAt = performance.now();
let loadingAnimationFrame = null;
let loadingTitleData = null;
let lastTitleProgressPercent = -1;
let environmentTransitionTimer = null;
let selectionTransitionTimer = null;
let interactionMode = "start";
let pointerX = ui.experience.clientWidth * 0.5;
let pointerY = ui.experience.clientHeight * 0.5;
let pointerTargetX = pointerX;
let pointerTargetY = pointerY;
let lastPointerRenderAt = performance.now();
let lastLightRenderAt = -Infinity;
let startGestureArmed = true;
let lastSelectionMouseAt = -Infinity;
let selectionMouseInside = false;
let selectionGestureArmed = false;
let selectionTargetLocked = false;
let selectionLockedLesson = null;
let selectionFistStartedAt = null;
let lastSelectionFistAt = -Infinity;
let tigerStepIndex = 0;
let tigerTransitionTimer = null;
let tigerPointerArmed = false;
let tigerLockedTarget = null;
let tigerFistStartedAt = null;
let lastTigerFistAt = -Infinity;
let swipeMode = null;
let swipeWarmup = null;
let swipeAnchor = null;
let swipeCooldownUntil = -Infinity;
let lastSwipeOpenAt = -Infinity;

const tigerStepNames = ["起势调息", "握拳上提", "虎爪上举", "握拳下拉", "回正"];
const tigerArtUrls = tigerStepNames.map((_, index) => `./assets/虎戏-动作-${index + 1}.png`);
const birdStepNames = ["准备站立", "两臂侧展", "舒展上举", "提膝平衡", "落臂还原"];
const birdArtUrls = birdStepNames.map((_, index) => `./assets/鸟戏-动作-${index + 1}.png`);
const deerStepNames = ["左脚上步", "重心前移", "立身画弧", "转腰下势", "两臂转动"];
const deerArtUrls = deerStepNames.map((_, index) => `./assets/鹿戏-动作-${index + 1}.png`);
const lessonDefinitions = {
  tiger: { name: "虎戏", stepNames: tigerStepNames, artUrls: tigerArtUrls },
  bird: { name: "鸟戏", stepNames: birdStepNames, artUrls: birdArtUrls },
  deer: { name: "鹿戏", stepNames: deerStepNames, artUrls: deerArtUrls },
};
let activeLesson = lessonDefinitions.tiger;
for (const src of [...tigerArtUrls, ...birdArtUrls, ...deerArtUrls]) {
  const image = new Image();
  image.src = src;
}

const gestureNames = {
  Closed_Fist: "已识别握拳",
  Open_Palm: "已检测手掌 · 请握拳",
  Pointing_Up: "请将五指完全握紧",
  Thumb_Down: "请将五指完全握紧",
  Thumb_Up: "请将拇指一并收拢",
  Victory: "请将五指完全握紧",
  ILoveYou: "请将五指完全握紧",
  None: "已检测到手 · 请握拳",
};

async function createRecognizer() {
  setStatus("正在加载手势模型", "首次加载需要连接网络");
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const options = (delegate) => ({
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate,
      },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.5,
      cannedGesturesClassifierOptions: {
        scoreThreshold: 0.45,
        maxResults: 1,
      },
    });

  try {
    recognizer = await GestureRecognizer.createFromOptions(vision, options("GPU"));
  } catch (gpuError) {
    console.warn("GPU delegate unavailable; falling back to CPU.", gpuError);
    recognizer = await GestureRecognizer.createFromOptions(vision, options("CPU"));
  }
}

async function enableCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("当前环境无法访问摄像头，请使用 localhost 或 HTTPS 打开页面");
  }

  setStatus("等待摄像头授权", "请在浏览器提示中选择“允许”");
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 30, max: 30 },
    },
  });

  ui.video.srcObject = stream;
  await new Promise((resolve) => {
    ui.video.addEventListener("loadeddata", resolve, { once: true });
  });
  await ui.video.play();
  resizeCanvas();
}

async function startCapture() {
  stopCapture();
  resetGestureProgress();
  ui.retryButton.hidden = true;

  try {
    await Promise.all([recognizer ? Promise.resolve() : createRecognizer(), enableCamera()]);
    running = true;
    ui.capturePanel.classList.add("is-ready");
    ui.fpsLabel.textContent = "实时";
    setStatus("未检测到手", "请将一只手完整放入识别框");
    requestAnimationFrame(predictFrame);
  } catch (error) {
    console.error(error);
    stopCapture();
    ui.capturePanel.classList.remove("is-ready", "is-holding");
    ui.fpsLabel.textContent = "未连接";
    setStatus("无法启用手部动捕", readableCameraError(error));
    ui.retryButton.hidden = false;
  }
}

function stopCapture() {
  running = false;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = null;
  }
  if (ui.video.srcObject) ui.video.srcObject = null;
}

function resizeCanvas() {
  const width = ui.video.videoWidth || 640;
  const height = ui.video.videoHeight || 480;
  if (ui.canvas.width !== width || ui.canvas.height !== height) {
    ui.canvas.width = width;
    ui.canvas.height = height;
    drawingUtils = new DrawingUtils(ui.canvas.getContext("2d"));
  }
}

function setPointerTarget(x, y) {
  pointerTargetX = x;
  pointerTargetY = y;
}

function mapHandToStage(point, stageRect, mode) {
  const range = POINTER_CAMERA_RANGE[mode];
  const horizontal = ((1 - point.x) - (1 - range.x) / 2) / range.x;
  const vertical = (point.y - (1 - range.y) / 2) / range.y;
  return {
    x: Math.max(18, Math.min(stageRect.width - 18, horizontal * stageRect.width)),
    y: Math.max(18, Math.min(stageRect.height - 18, vertical * stageRect.height)),
  };
}

function freezePointer() {
  setPointerTarget(pointerX, pointerY);
}

function centerPointer() {
  pointerX = ui.experience.clientWidth / 2;
  pointerY = ui.experience.clientHeight / 2;
  setPointerTarget(pointerX, pointerY);
}

function animatePointer(now) {
  const frameRatio = Math.min(2.5, Math.max(0, (now - lastPointerRenderAt) / 16.7));
  lastPointerRenderAt = now;
  const blend = 1 - Math.pow(1 - POINTER_SMOOTHING, frameRatio);
  pointerX += (pointerTargetX - pointerX) * blend;
  pointerY += (pointerTargetY - pointerY) * blend;
  if (ui.gesturePointer.classList.contains("is-visible")) {
    ui.gesturePointer.style.transform = `translate3d(${pointerX - 22}px, ${pointerY - 22}px, 0)`;
    if (interactionMode === "selection" && ui.selectionScreen.classList.contains("has-light") && now - lastLightRenderAt >= 32) {
      ui.selectionScreen.style.setProperty("--light-x", `${pointerX}px`);
      ui.selectionScreen.style.setProperty("--light-y", `${pointerY}px`);
      lastLightRenderAt = now;
    }
  }
  requestAnimationFrame(animatePointer);
}

function predictFrame(now) {
  if (!running) return;
  requestAnimationFrame(predictFrame);

  if (
    !recognizer ||
    ui.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
    ui.video.currentTime === lastVideoTime ||
    now - lastInferenceAt < INFERENCE_INTERVAL_MS
  ) {
    return;
  }

  lastVideoTime = ui.video.currentTime;
  resizeCanvas();

  const result = recognizer.recognizeForVideo(ui.video, now);
  // Leave a short gap after slow inference so the browser can paint cursor frames.
  lastInferenceAt = Math.max(now, performance.now() - (INFERENCE_INTERVAL_MS - 18));
  renderLandmarks(result.landmarks ?? []);
  if (updateBackSwipe(result, now)) {
    updateFps(now);
    return;
  }
  if (interactionMode === "start") updateGesture(result, now);
  else if (interactionMode === "loading") updatePointerInteraction(result);
  else if (interactionMode === "selection") updateSelectionInteraction(result, now);
  else if (interactionMode === "tiger") updateTigerInteraction(result, now);
  updateFps(now);
}

function updateBackSwipe(result, now) {
  if (swipeMode !== interactionMode) {
    swipeMode = interactionMode;
    swipeWarmup = null;
    swipeAnchor = null;
    lastSwipeOpenAt = -Infinity;
  }
  if (!["loading", "selection", "tiger"].includes(interactionMode)) return false;

  const palm = result.landmarks?.[0]?.[9];
  const gesture = result.gestures?.[0]?.[0];
  const openPalmRecognized = gesture?.categoryName === "Open_Palm" && gesture.score >= 0.55;
  if (openPalmRecognized) lastSwipeOpenAt = now;
  const openPalmTracked = openPalmRecognized ||
    (gesture?.categoryName !== "Closed_Fist" && now - lastSwipeOpenAt <= FLICKER_GRACE_MS);
  if (!palm || !openPalmTracked) {
    swipeWarmup = null;
    swipeAnchor = null;
    return false;
  }
  if (now < swipeCooldownUntil) return false;

  const point = { x: 1 - palm.x, y: palm.y, at: now };
  if (!swipeWarmup) {
    swipeWarmup = point;
    return false;
  }
  if (!swipeAnchor) {
    if (Math.hypot(point.x - swipeWarmup.x, point.y - swipeWarmup.y) > 0.045) {
      swipeWarmup = point;
    } else if (now - swipeWarmup.at >= SWIPE_STILL_MS) {
      swipeAnchor = point;
    }
    return false;
  }

  const elapsed = now - swipeAnchor.at;
  const dx = point.x - swipeAnchor.x;
  const dy = Math.abs(point.y - swipeAnchor.y);
  if (elapsed > SWIPE_MAX_MS || dy > 0.13 || dx > 0.06) {
    swipeAnchor = point;
    return false;
  }
  if (elapsed < 90 || dx > -SWIPE_LEFT_DISTANCE || dy > Math.abs(dx) * 0.4) return false;

  swipeCooldownUntil = now + SWIPE_COOLDOWN_MS;
  swipeWarmup = null;
  swipeAnchor = null;
  setPointerProgress(0);
  setCaptureConfirmation(0);
  setStatus("手掌左滑", "正在后退");

  if (interactionMode === "tiger") {
    tigerPointerArmed = false;
    tigerLockedTarget = null;
    tigerFistStartedAt = null;
    if (tigerStepIndex > 0) setTigerStep(tigerStepIndex - 1);
    else returnToSelection();
  } else {
    returnToStart();
  }
  return true;
}

function renderLandmarks(hands) {
  const context = ui.canvas.getContext("2d");
  context.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
  if (!drawingUtils) drawingUtils = new DrawingUtils(context);

  for (const landmarks of hands) {
    drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
      color: "rgba(232, 194, 119, 0.82)",
      lineWidth: 3,
    });
    drawingUtils.drawLandmarks(landmarks, {
      color: "#f1dfba",
      fillColor: "rgba(112, 54, 35, 0.9)",
      lineWidth: 1.5,
      radius: 3.5,
    });
  }
}

function updateGesture(result, now) {
  const hasHand = Boolean(result.landmarks?.length);
  const category = result.gestures?.[0]?.[0];
  const gestureName = category?.categoryName ?? "None";
  const gestureScore = category?.score ?? 0;
  const isClosedFist = gestureName === "Closed_Fist" && gestureScore >= CLOSED_FIST_THRESHOLD;

  if (!hasHand) {
    resetGestureProgress();
    setStatus("未检测到手", "请将一只手完整放入识别框");
    return;
  }

  if (!isClosedFist) startGestureArmed = true;
  if (isClosedFist && !startGestureArmed) {
    resetGestureProgress();
    setStatus("请先松开手掌", "张开手掌后，再握拳开始");
    return;
  }

  if (isClosedFist) lastClosedFistAt = now;
  const withinGrace = now - lastClosedFistAt <= FLICKER_GRACE_MS;

  if (isClosedFist || (fistStartedAt !== null && withinGrace)) {
    if (fistStartedAt === null) fistStartedAt = now;
    const progress = Math.min(1, (now - fistStartedAt) / HOLD_DURATION_MS);
    setProgress(progress);
    ui.capturePanel.classList.add("is-holding");
    setStatus("保持握拳", `稳定保持 ${Math.ceil((HOLD_DURATION_MS - progress * HOLD_DURATION_MS) / 100) / 10} 秒`);

    if (progress >= 1) triggerStart();
    return;
  }

  resetGestureProgress();
  setStatus(gestureNames[gestureName] ?? "已检测到手 · 请握拳", "五指完全收拢后保持片刻");
}

function setProgress(progress) {
  const safeProgress = Math.max(0, Math.min(1, progress));
  ui.progressRing.style.setProperty("--progress", safeProgress.toFixed(3));
  ui.progressValue.textContent = Math.round(safeProgress * 100);
}

function setCaptureConfirmation(progress) {
  setProgress(progress);
  ui.capturePanel.classList.toggle("is-holding", progress > 0);
}

function resetGestureProgress() {
  fistStartedAt = null;
  lastClosedFistAt = -Infinity;
  ui.capturePanel.classList.remove("is-holding");
  setProgress(0);
}

function triggerStart() {
  if (triggered) return;
  triggered = true;
  interactionMode = "transition";
  setProgress(1);
  setStatus("识别成功", "正在进入五禽戏");
  ui.fpsLabel.textContent = "已触发";
  ui.startScreen.classList.add("is-triggered");
  beginEnvironmentTransition();

  window.setTimeout(() => {
    ui.loadingScreen.classList.add("is-active");
    ui.startScreen.classList.remove("is-active");
    interactionMode = "loading";
    centerPointer();
    resetGestureProgress();
    setStatus("手部动捕已连接", "张掌稍停 · 快速左滑返回");
    startLoadingAnimation();
  }, 520);
}

function beginEnvironmentTransition() {
  if (environmentTransitionTimer !== null) clearTimeout(environmentTransitionTimer);
  ui.experience.classList.remove("is-transitioning");
  void ui.experience.offsetWidth;
  ui.experience.classList.add("is-transitioning");
  environmentTransitionTimer = window.setTimeout(() => {
    ui.experience.classList.remove("is-transitioning");
    environmentTransitionTimer = null;
  }, 1450);
}

function updatePointerInteraction(result) {
  const landmarks = result.landmarks?.[0];
  const category = result.gestures?.[0]?.[0];
  const isClosedFist =
    category?.categoryName === "Closed_Fist" && category.score >= CLOSED_FIST_THRESHOLD;

  if (!landmarks) {
    ui.gesturePointer.classList.remove("is-visible", "is-pressing", "is-locked");
    ui.backButton.classList.remove("is-gesture-hover");
    setPointerStatus("请将手掌放入摄像头范围", false);
    setPointerProgress(0);
    return;
  }

  if (!isClosedFist) {
    const stageRect = ui.experience.getBoundingClientRect();
    const target = mapHandToStage(landmarks[8], stageRect, "loading");
    setPointerTarget(target.x, target.y);
  } else {
    freezePointer();
  }

  ui.gesturePointer.style.transform = `translate3d(${pointerX - 22}px, ${pointerY - 22}px, 0)`;
  ui.gesturePointer.classList.add("is-visible");
  ui.gesturePointer.classList.remove("is-pressing", "is-locked");
  ui.backButton.classList.remove("is-gesture-hover");
  setPointerProgress(0);
  setPointerStatus("张开手掌稍停 · 快速左滑返回");
}

function updateSelectionInteraction(result, now) {
  const landmarks = result.landmarks?.[0];
  if (selectionMouseInside && (!landmarks || now - lastSelectionMouseAt < 1400)) return;
  if (!landmarks) {
    clearSelectionLight();
    selectionGestureArmed = false;
    selectionTargetLocked = false;
    selectionLockedLesson = null;
    selectionFistStartedAt = null;
    setPointerProgress(0);
    setCaptureConfirmation(0);
    setStatus("未检测到手", "请将手掌放入摄像头范围");
    return;
  }

  const category = result.gestures?.[0]?.[0];
  const isClosedFist = category?.categoryName === "Closed_Fist" && category.score >= CLOSED_FIST_THRESHOLD;
  if (isClosedFist) lastSelectionFistAt = now;
  const fistTracked = isClosedFist || (selectionTargetLocked && now - lastSelectionFistAt < FLICKER_GRACE_MS);
  const indexAngle = fingerJointAngle(landmarks[5], landmarks[6], landmarks[8]);
  const fingerClosing = indexAngle < 158;
  if (!fistTracked && indexAngle > 165) {
    selectionGestureArmed = true;
    selectionTargetLocked = false;
    selectionLockedLesson = null;
    selectionFistStartedAt = null;
    setPointerProgress(0);
    setCaptureConfirmation(0);
    ui.gesturePointer.classList.remove("is-pressing", "is-locked");
  }

  // Use the middle knuckle near the palm center, so an open hand points steadily.
  if (!selectionTargetLocked && !fingerClosing && !isClosedFist) {
    const palm = landmarks[9];
    const stageRect = ui.experience.getBoundingClientRect();
    const target = mapHandToStage(palm, stageRect, "selection");
    setPointerTarget(target.x, target.y);
  } else if (fingerClosing || isClosedFist) {
    freezePointer();
  }
  const activeBadge = renderSelectionLight();
  if (selectionGestureArmed && activeBadge?.dataset.lesson && (fingerClosing || isClosedFist)) {
    selectionTargetLocked = true;
    selectionLockedLesson = activeBadge.dataset.lesson;
  }
  ui.gesturePointer.classList.toggle("is-locked", selectionTargetLocked);
  if (!selectionTargetLocked || !fistTracked) {
    selectionFistStartedAt = null;
    ui.gesturePointer.classList.remove("is-pressing");
    setPointerProgress(0);
    setCaptureConfirmation(0);
    return;
  }

  if (selectionFistStartedAt === null) selectionFistStartedAt = now;
  const progress = Math.min(1, (now - selectionFistStartedAt) / POINTER_CLICK_DURATION_MS);
  ui.gesturePointer.classList.add("is-pressing");
  const selectedName = lessonDefinitions[selectionLockedLesson]?.name ?? "五禽戏";
  ui.selectionHint.textContent = `保持握拳 · 进入${selectedName}`;
  setPointerProgress(progress);
  setCaptureConfirmation(progress);
  setStatus("保持握拳", `正在确认${selectedName}选择`);
  if (progress >= 1) {
    selectionGestureArmed = false;
    enterLesson(selectionLockedLesson);
  }
}

function renderSelectionLight(source = "hand") {
  const stageRect = ui.experience.getBoundingClientRect();
  ui.gesturePointer.style.transform = `translate3d(${pointerX - 22}px, ${pointerY - 22}px, 0)`;
  ui.gesturePointer.classList.add("is-visible");
  ui.selectionScreen.classList.add("has-light");

  const viewportX = stageRect.left + pointerX;
  const viewportY = stageRect.top + pointerY;
  let activeBadge = null;
  for (const badge of ui.selectionBadges) {
    const rect = badge.getBoundingClientRect();
    const dx = (viewportX - (rect.left + rect.width / 2)) / (rect.width / 2);
    const dy = (viewportY - (rect.top + rect.height / 2)) / (rect.height / 2);
    const hovered = dx * dx + dy * dy <= 1;
    badge.classList.toggle("is-hovered", hovered);
    if (hovered) activeBadge = badge;
  }

  ui.selectionHint.textContent = activeBadge?.dataset.lesson
    ? `已触及${activeBadge.dataset.name} · 握拳进入`
    : activeBadge ? `已触及${activeBadge.dataset.name}` : "手掌移动探索 · 稍停后左滑返回";
  setStatus(source === "mouse" ? "光点预览" : "手部动捕已连接", activeBadge ? `${activeBadge.dataset.name}徽章已高亮` : "移动手掌探索 · 左滑返回");
  return activeBadge;
}

function clearSelectionLight() {
  ui.gesturePointer.classList.remove("is-visible", "is-pressing", "is-locked");
  ui.selectionScreen.classList.remove("has-light");
  ui.selectionScreen.style.removeProperty("--light-x");
  ui.selectionScreen.style.removeProperty("--light-y");
  for (const badge of ui.selectionBadges) badge.classList.remove("is-hovered");
  ui.selectionHint.textContent = "手掌移动探索 · 稍停后左滑返回";
  setCaptureConfirmation(0);
}

function enterSelection() {
  selectionTransitionTimer = null;
  if (interactionMode !== "loading") return;
  interactionMode = "transition";
  ui.gesturePointer.classList.remove("is-visible", "is-pressing", "is-locked");
  ui.backButton.classList.remove("is-gesture-hover");
  beginEnvironmentTransition();
  window.setTimeout(() => {
    clearSelectionLight();
    selectionMouseInside = false;
    ui.selectionScreen.classList.add("is-active");
    ui.loadingScreen.classList.remove("is-active");
    ui.experience.classList.add("is-light-mode");
    interactionMode = "selection";
    centerPointer();
    setStatus("手部动捕已连接", "移动手掌探索 · 左滑返回");
  }, 520);
}

function setTigerStep(index) {
  const nextIndex = Math.max(0, Math.min(activeLesson.stepNames.length - 1, index));
  if (nextIndex === tigerStepIndex && ui.tigerScreen.classList.contains("is-active")) return;
  finishTigerCrossfade();
  tigerStepIndex = nextIndex;
  const nextSrc = activeLesson.artUrls[tigerStepIndex];
  if (ui.tigerScreen.classList.contains("is-active")) {
    ui.tigerArtIncoming.style.transition = "none";
    ui.tigerArtIncoming.style.opacity = "0";
    ui.tigerArtIncoming.src = nextSrc;
    void ui.tigerArtIncoming.offsetWidth;
    ui.tigerArtIncoming.style.transition = "";
    ui.tigerArtIncoming.style.opacity = "";
    ui.tigerScreen.classList.add("is-switching");
    tigerTransitionTimer = window.setTimeout(finishTigerCrossfade, 500);
  } else {
    ui.tigerArt.src = nextSrc;
    ui.tigerArtIncoming.src = nextSrc;
  }
  ui.tigerArt.alt = `${activeLesson.name}动作学习，第${tigerStepIndex + 1}步：${activeLesson.stepNames[tigerStepIndex]}`;
  const finalStep = tigerStepIndex === activeLesson.stepNames.length - 1;
  ui.tigerScreen.classList.toggle("is-final-step", finalStep);
  ui.tigerNext.setAttribute("aria-label", finalStep ? "握拳完成" : "握拳前进");
  for (const button of ui.tigerSteps) {
    const step = Number(button.dataset.step);
    button.setAttribute("aria-label", `查看第${step + 1}步：${activeLesson.stepNames[step]}`);
    if (step === tigerStepIndex) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  }
  if (interactionMode === "tiger") {
    setStatus(`${activeLesson.name}动作学习`, `第${tigerStepIndex + 1}步 · 握拳前进 / 左滑后退`);
  }
}

function finishTigerCrossfade() {
  if (tigerTransitionTimer !== null) {
    clearTimeout(tigerTransitionTimer);
    tigerTransitionTimer = null;
  }
  if (ui.tigerScreen.classList.contains("is-switching")) {
    ui.tigerArt.src = ui.tigerArtIncoming.src;
    ui.tigerScreen.classList.remove("is-switching");
  }
}

function enterLesson(type) {
  if (interactionMode !== "selection" || !lessonDefinitions[type]) return;
  activeLesson = lessonDefinitions[type];
  interactionMode = "transition";
  selectionGestureArmed = false;
  selectionTargetLocked = false;
  selectionLockedLesson = null;
  selectionFistStartedAt = null;
  clearSelectionLight();
  setPointerProgress(0);
  beginEnvironmentTransition();
  window.setTimeout(() => {
    ui.tigerScreen.setAttribute("aria-label", `${activeLesson.name}动作学习界面`);
    setTigerStep(0);
    ui.tigerScreen.classList.add("is-active");
    ui.selectionScreen.classList.remove("is-active");
    ui.experience.classList.remove("is-light-mode");
    ui.experience.classList.add("is-tiger-mode");
    interactionMode = "tiger";
    tigerPointerArmed = false;
    tigerLockedTarget = null;
    tigerFistStartedAt = null;
    centerPointer();
    setStatus(`${activeLesson.name}动作学习`, "握拳前进 · 张掌左滑后退");
  }, 520);
}

function returnToSelection() {
  if (interactionMode !== "tiger") return;
  finishTigerCrossfade();
  interactionMode = "transition";
  ui.gesturePointer.classList.remove("is-visible", "is-pressing", "is-locked");
  for (const target of ui.tigerTargets) target.classList.remove("is-gesture-hover");
  beginEnvironmentTransition();
  window.setTimeout(() => {
    ui.selectionScreen.classList.add("is-active");
    ui.tigerScreen.classList.remove("is-active");
    ui.experience.classList.remove("is-tiger-mode");
    ui.experience.classList.add("is-light-mode");
    clearSelectionLight();
    selectionMouseInside = false;
    selectionGestureArmed = false;
    interactionMode = "selection";
    centerPointer();
    setStatus("已返回五禽选择", "移动手掌探索 · 左滑返回");
  }, 520);
}

function nextTigerStep() {
  if (interactionMode !== "tiger") return;
  if (tigerStepIndex < activeLesson.stepNames.length - 1) setTigerStep(tigerStepIndex + 1);
  else returnToSelection();
}

function tigerTargetAt(viewportX, viewportY, padding = 0) {
  return ui.tigerTargets.find((target) => {
    if (target === ui.tigerBack) return false;
    const rect = target.getBoundingClientRect();
    return viewportX >= rect.left - padding && viewportX <= rect.right + padding &&
      viewportY >= rect.top - padding && viewportY <= rect.bottom + padding;
  });
}

function updateTigerInteraction(result, now) {
  const landmarks = result.landmarks?.[0];
  if (!landmarks) {
    ui.gesturePointer.classList.remove("is-visible", "is-pressing", "is-locked");
    for (const target of ui.tigerTargets) target.classList.remove("is-gesture-hover");
    tigerPointerArmed = false;
    tigerLockedTarget = null;
    tigerFistStartedAt = null;
    setPointerProgress(0);
    setCaptureConfirmation(0);
    return;
  }

  const category = result.gestures?.[0]?.[0];
  const isClosedFist = category?.categoryName === "Closed_Fist" && category.score >= CLOSED_FIST_THRESHOLD;
  if (isClosedFist) lastTigerFistAt = now;
  const indexAngle = fingerJointAngle(landmarks[5], landmarks[6], landmarks[8]);
  const fingerExtended = indexAngle > 165;
  const fingerClosing = indexAngle < 158;
  const fingerReleased = !isClosedFist && now - lastTigerFistAt > FLICKER_GRACE_MS &&
    (fingerExtended || category?.categoryName === "Open_Palm" || category?.categoryName === "Pointing_Up");
  if (fingerReleased) {
    tigerPointerArmed = true;
    tigerLockedTarget = null;
    tigerFistStartedAt = null;
    setPointerProgress(0);
    setCaptureConfirmation(0);
    ui.gesturePointer.classList.remove("is-pressing", "is-locked");
  }

  const stageRect = ui.experience.getBoundingClientRect();
  const nearTarget = tigerTargetAt(stageRect.left + pointerX, stageRect.top + pointerY);
  const shouldFreeze = isClosedFist || (fingerClosing && tigerPointerArmed && nearTarget);
  if (!tigerLockedTarget && !shouldFreeze) {
    const tip = landmarks[8];
    const target = mapHandToStage(tip, stageRect, "tiger");
    setPointerTarget(target.x, target.y);
  } else if (shouldFreeze) {
    freezePointer();
  }
  ui.gesturePointer.style.transform = `translate3d(${pointerX - 22}px, ${pointerY - 22}px, 0)`;
  ui.gesturePointer.classList.add("is-visible");

  const viewportX = stageRect.left + pointerX;
  const viewportY = stageRect.top + pointerY;
  const hovered = tigerTargetAt(viewportX, viewportY);
  if (tigerPointerArmed && hovered && (isClosedFist || (fingerClosing && nearTarget === hovered))) {
    tigerLockedTarget = hovered;
  } else if (tigerPointerArmed && isClosedFist && !tigerLockedTarget) {
    tigerLockedTarget = ui.tigerNext;
  }
  for (const target of ui.tigerTargets) target.classList.toggle("is-gesture-hover", target === (tigerLockedTarget || hovered));
  ui.gesturePointer.classList.toggle("is-locked", Boolean(tigerLockedTarget));

  const fistTracked = isClosedFist || (tigerLockedTarget && now - lastTigerFistAt < FLICKER_GRACE_MS);
  if (!tigerLockedTarget || !fistTracked) {
    tigerFistStartedAt = null;
    ui.gesturePointer.classList.remove("is-pressing");
    setPointerProgress(0);
    setCaptureConfirmation(0);
    return;
  }

  if (tigerFistStartedAt === null) tigerFistStartedAt = now;
  const progress = Math.min(1, (now - tigerFistStartedAt) / POINTER_CLICK_DURATION_MS);
  ui.gesturePointer.classList.add("is-pressing");
  setPointerProgress(progress);
  setCaptureConfirmation(progress);
  if (progress >= 1) {
    const target = tigerLockedTarget;
    tigerPointerArmed = false;
    tigerLockedTarget = null;
    tigerFistStartedAt = null;
    setCaptureConfirmation(0);
    target.click();
  }
}

function fingerJointAngle(base, joint, tip) {
  const ax = base.x - joint.x;
  const ay = base.y - joint.y;
  const az = base.z - joint.z;
  const bx = tip.x - joint.x;
  const by = tip.y - joint.y;
  const bz = tip.z - joint.z;
  const dot = ax * bx + ay * by + az * bz;
  const lengths = Math.hypot(ax, ay, az) * Math.hypot(bx, by, bz);
  return lengths ? Math.acos(Math.max(-1, Math.min(1, dot / lengths))) * 180 / Math.PI : 180;
}

function setPointerProgress(progress) {
  const safeProgress = Math.max(0, Math.min(1, progress));
  ui.gesturePointer.style.setProperty("--pointer-progress", safeProgress.toFixed(3));
  ui.gesturePointer.style.setProperty("--pointer-angle", `${safeProgress * 360}deg`);
}

function returnToStart() {
  if (interactionMode === "transition") return;

  if (loadingAnimationFrame !== null) {
    cancelAnimationFrame(loadingAnimationFrame);
    loadingAnimationFrame = null;
  }
  if (selectionTransitionTimer !== null) {
    clearTimeout(selectionTransitionTimer);
    selectionTransitionTimer = null;
  }

  interactionMode = "transition";
  ui.gesturePointer.classList.remove("is-visible", "is-pressing", "is-locked");
  ui.backButton.classList.remove("is-gesture-hover");
  beginEnvironmentTransition();

  window.setTimeout(() => {
    interactionMode = "start";
    triggered = false;
    startGestureArmed = false;
    ui.loadingScreen.classList.remove("is-active", "is-loading", "is-complete");
    ui.selectionScreen.classList.remove("is-active");
    ui.experience.classList.remove("is-light-mode");
    selectionMouseInside = false;
    clearSelectionLight();
    ui.startScreen.classList.remove("is-triggered");
    ui.startScreen.classList.add("is-active");
    ui.fpsLabel.textContent = "实时";
    setLoadingProgress(0);
    resetGestureProgress();
    setStatus("已返回开始界面", "张开手掌后，握拳即可重新开始");
  }, 520);
}

function startLoadingAnimation() {
  if (loadingAnimationFrame !== null) cancelAnimationFrame(loadingAnimationFrame);

  ui.loadingScreen.classList.remove("is-complete");
  ui.loadingScreen.classList.add("is-loading");
  setLoadingProgress(0);
  const startedAt = performance.now();

  function updateLoading(now) {
    const elapsedRatio = Math.min(1, (now - startedAt) / LOADING_DURATION_MS);
    const progress = stagedLoadingProgress(elapsedRatio);
    setLoadingProgress(progress);

    if (elapsedRatio < 1) {
      loadingAnimationFrame = requestAnimationFrame(updateLoading);
      return;
    }

    loadingAnimationFrame = null;
    ui.loadingScreen.classList.remove("is-loading");
    ui.loadingScreen.classList.add("is-complete");
    ui.loadingPhase.textContent = "加载完成";
    selectionTransitionTimer = window.setTimeout(enterSelection, 1100);
    window.setTimeout(() => {
      window.dispatchEvent(new CustomEvent("wuqinxi:ready"));
    }, 700);
  }

  loadingAnimationFrame = requestAnimationFrame(updateLoading);
}

async function prepareLoadingTitle() {
  const source = new Image();
  const base = new Image();
  source.src = "./assets/五禽戏墨韵标题-加载初始态.png";
  base.src = "./assets/五禽戏-华佗墨韵加载中.png";
  await Promise.all([source.decode(), base.decode()]);

  // Align the supplied three-character artwork over the loading page.
  // Only its ink pixels participate in the progress-driven reveal.
  const destination = { x: 267, y: 121, width: 1110, height: 445 };
  const scratch = document.createElement("canvas");
  scratch.width = destination.width;
  scratch.height = destination.height;
  const scratchContext = scratch.getContext("2d", { willReadFrequently: true });
  scratchContext.drawImage(source, 250, 180, 1180, 480, 0, 0, destination.width, destination.height);
  const title = scratchContext.getImageData(0, 0, destination.width, destination.height);
  scratchContext.clearRect(0, 0, destination.width, destination.height);
  scratchContext.drawImage(base, destination.x, destination.y, destination.width, destination.height, 0, 0, destination.width, destination.height);
  const oldTitle = scratchContext.getImageData(0, 0, destination.width, destination.height);

  for (let i = 0; i < title.data.length; i += 4) {
    const red = title.data[i];
    const green = title.data[i + 1];
    const blue = title.data[i + 2];
    const brightness = (red + green + blue) / 3;
    const isSealRed = red > green * 1.35 && red > blue * 1.35;
    title.data[i + 3] = isSealRed ? 0 : Math.round(Math.max(0, Math.min(1, (brightness - 42) / 100)) * 255);

    const oldRed = oldTitle.data[i];
    const oldGreen = oldTitle.data[i + 1];
    const oldBlue = oldTitle.data[i + 2];
    const oldBrightness = (oldRed + oldGreen + oldBlue) / 3;
    const oldIsSealRed = oldRed > oldGreen * 1.35 && oldRed > oldBlue * 1.35;
    oldTitle.data[i] = 6;
    oldTitle.data[i + 1] = 6;
    oldTitle.data[i + 2] = 5;
    oldTitle.data[i + 3] = oldIsSealRed ? 0 : Math.round(Math.max(0, Math.min(1, (oldBrightness - 42) / 95)) * 250);
  }

  const canvas = ui.loadingTitleCanvas;
  canvas.width = 1672;
  canvas.height = 941;
  const context = canvas.getContext("2d");
  context.putImageData(oldTitle, destination.x, destination.y);
  loadingTitleData = {
    destination,
    source: title,
    current: scratchContext.createImageData(destination.width, destination.height),
    context,
  };
  lastTitleProgressPercent = -1;
  renderLoadingTitle(0);
}

function smoothstep(value) {
  const t = Math.max(0, Math.min(1, value));
  return t * t * (3 - 2 * t);
}

function renderLoadingTitle(progress) {
  if (!loadingTitleData) return;
  const percent = Math.round(progress * 100);
  if (percent === lastTitleProgressPercent) return;
  lastTitleProgressPercent = percent;

  const { destination, source, current, context } = loadingTitleData;
  const from = source.data;
  const to = current.data;
  const progressValue = percent / 100;

  for (let y = 0; y < destination.height; y++) {
    for (let x = 0; x < destination.width; x++) {
      const i = (y * destination.width + x) * 4;
      if (from[i + 3] === 0) {
        to[i + 3] = 0;
        continue;
      }
      // The leading edge uses the same 0–100% position as the progress bar.
      const grain = ((x * 73 + y * 137) % 97) / 97;
      const inkPosition = (x + 0.5) / destination.width;
      const reveal = progressValue === 0
        ? 0
        : progressValue === 1
          ? 1
          : smoothstep((progressValue - inkPosition + 0.024 + (grain - 0.5) * 0.008) / 0.048);
      to[i] = 49 + (from[i] - 49) * reveal;
      to[i + 1] = 48 + (from[i + 1] - 48) * reveal;
      to[i + 2] = 44 + (from[i + 2] - 44) * reveal;
      to[i + 3] = from[i + 3];
    }
  }
  context.putImageData(current, destination.x, destination.y);
}

function stagedLoadingProgress(time) {
  if (time < 0.18) return (time / 0.18) * 0.22;
  if (time < 0.55) return 0.22 + ((time - 0.18) / 0.37) * 0.42;
  if (time < 0.86) return 0.64 + ((time - 0.55) / 0.31) * 0.25;
  return 0.89 + ((time - 0.86) / 0.14) * 0.11;
}

function setLoadingProgress(progress) {
  const safeProgress = Math.max(0, Math.min(1, progress));
  const percent = Math.round(safeProgress * 100);
  const displayedProgress = percent / 100;
  ui.loadingMotion.style.setProperty("--load-progress", displayedProgress.toFixed(2));
  ui.loadingMotion.style.setProperty("--load-position", `${percent}%`);
  ui.loadingFill.style.transform = `scaleX(${displayedProgress})`;
  ui.loadingPercent.textContent = `${percent}%`;
  renderLoadingTitle(displayedProgress);

  if (percent < 22) ui.loadingPhase.textContent = "调息凝神";
  else if (percent < 64) ui.loadingPhase.textContent = "导引五禽";
  else if (percent < 89) ui.loadingPhase.textContent = "墨韵成形";
  else if (percent < 100) ui.loadingPhase.textContent = "即将完成";
}

function setStatus(primary, secondary) {
  ui.gestureLabel.textContent = primary;
  ui.statusLabel.textContent = secondary;
}

function setPointerStatus(message, hasHand = true) {
  ui.pointerStatus.textContent = message;
  setStatus(hasHand ? "手部动捕已连接" : "未检测到手", message);
}

function updateFps(now) {
  frameCounter += 1;
  const elapsed = now - fpsWindowStartedAt;
  if (elapsed >= 1000) {
    ui.fpsLabel.textContent = `${Math.round((frameCounter * 1000) / elapsed)} FPS`;
    frameCounter = 0;
    fpsWindowStartedAt = now;
  }
}

function readableCameraError(error) {
  if (error?.name === "NotAllowedError") return "摄像头权限被拒绝，请在浏览器地址栏中重新允许";
  if (error?.name === "NotFoundError") return "未找到可用摄像头";
  if (error?.name === "NotReadableError") return "摄像头正被其他程序占用";
  return error?.message || "请刷新页面后重试";
}

ui.retryButton.addEventListener("click", startCapture);
ui.backButton.addEventListener("click", returnToStart);
document.querySelector(".badge-tiger").addEventListener("click", () => enterLesson("tiger"));
document.querySelector(".badge-bird").addEventListener("click", () => enterLesson("bird"));
document.querySelector(".badge-deer").addEventListener("click", () => enterLesson("deer"));
ui.tigerBack.addEventListener("click", returnToSelection);
ui.tigerNext.addEventListener("click", nextTigerStep);
for (const button of ui.tigerSteps) {
  button.addEventListener("click", () => {
    if (interactionMode === "tiger") setTigerStep(Number(button.dataset.step));
  });
}
ui.selectionScreen.addEventListener("mousemove", (event) => {
  if (interactionMode !== "selection") return;
  const rect = ui.experience.getBoundingClientRect();
  selectionMouseInside = true;
  lastSelectionMouseAt = performance.now();
  pointerX = Math.max(18, Math.min(rect.width - 18, event.clientX - rect.left));
  pointerY = Math.max(18, Math.min(rect.height - 18, event.clientY - rect.top));
  setPointerTarget(pointerX, pointerY);
  renderSelectionLight("mouse");
});
ui.selectionScreen.addEventListener("mouseleave", () => {
  if (interactionMode !== "selection") return;
  selectionMouseInside = false;
  lastSelectionMouseAt = -Infinity;
  clearSelectionLight();
});
window.addEventListener("resize", resizeCanvas);
window.addEventListener("pagehide", stopCapture);

prepareLoadingTitle().catch((error) => console.warn("标题素材加载失败", error));
requestAnimationFrame(animatePointer);
startCapture();
