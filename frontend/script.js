/* ==========================================================================
   AI Exam Proctoring System — Shared JS
   ProctorEngine (webcam + alert + trust score) · ExamController (timer/palette)
   No server dependency. Face/pose detection here is heuristic (brightness +
   motion + frame-diff based) — swap in a real CV model if wiring to backend.
   ========================================================================== */

/* ---------------------------------- Utils ---------------------------------- */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function pad(n) { return n.toString().padStart(2, '0'); }

function nowStamp() {
  const d = new Date();
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function uid() { return Math.random().toString(36).slice(2, 9); }

/* ---------------------------------- ProctorEngine ---------------------------------- */
/**
 * Handles webcam access, heuristic behavioral checks, alert logging,
 * and running trust score. Emits 'alert' and 'score' events.
 */
class ProctorEngine extends EventTarget {
  constructor(opts = {}) {
    super();
    this.video = opts.video || null;         // <video> element
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });

    this.interval = opts.interval || 1500;    // ms between checks
    this.trustScore = 100;
    this.alerts = [];
    this.stream = null;
    this.timer = null;
    this.prevFrame = null;

    this.thresholds = {
      darkness: opts.darkness ?? 25,          // avg luma below = no-face/camera covered
      motionLow: opts.motionLow ?? 0.4,        // % pixel diff below = suspiciously static (tampering)
      motionHigh: opts.motionHigh ?? 35,       // % pixel diff above = excessive movement (multi-person/away)
    };

    this.penalties = {
      no_face: 8,
      multiple_faces: 12,
      looking_away: 5,
      camera_tamper: 15,
      tab_switch: 10,
    };
  }

  async start() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      if (this.video) {
        this.video.srcObject = this.stream;
        await this.video.play();
        this.canvas.width = this.video.videoWidth || 320;
        this.canvas.height = this.video.videoHeight || 240;
      }
      this.timer = setInterval(() => this._tick(), this.interval);
      this._bindTabVisibility();
      return true;
    } catch (err) {
      this._raise('camera_tamper', 'Camera access denied or unavailable', 'high');
      return false;
    }
  }

  stop() {
    clearInterval(this.timer);
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    document.removeEventListener('visibilitychange', this._visHandler);
  }

  _bindTabVisibility() {
    this._visHandler = () => {
      if (document.hidden) {
        this._raise('tab_switch', 'Tab switched or window lost focus', 'medium');
      }
    };
    document.addEventListener('visibilitychange', this._visHandler);
  }

  _tick() {
    if (!this.video || this.video.readyState < 2) return;

    this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
    const frame = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    const luma = this._avgLuma(frame);
    const motion = this.prevFrame ? this._frameDiff(frame, this.prevFrame) : 0;
    this.prevFrame = frame;

    if (luma < this.thresholds.darkness) {
      this._raise('no_face', 'No face detected — frame too dark or camera covered', 'high');
      return;
    }

    if (motion < this.thresholds.motionLow) {
      this._raise('camera_tamper', 'Static frame detected — possible camera tampering', 'high');
      return;
    }

    if (motion > this.thresholds.motionHigh) {
      this._raise('looking_away', 'Excessive movement — candidate may be looking away', 'medium');
      return;
    }

    // Randomized low-rate simulation of multi-face detection for demo purposes.
    if (Math.random() < 0.015) {
      this._raise('multiple_faces', 'Additional face detected in frame', 'high');
    }
  }

  _avgLuma(frame) {
    const d = frame.data;
    let sum = 0;
    const step = 16; // sample for performance
    let count = 0;
    for (let i = 0; i < d.length; i += 4 * step) {
      sum += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      count++;
    }
    return sum / count;
  }

  _frameDiff(a, b) {
    const da = a.data, db = b.data;
    let diff = 0;
    const step = 16;
    let count = 0;
    for (let i = 0; i < da.length; i += 4 * step) {
      diff += Math.abs(da[i] - db[i]);
      count++;
    }
    return (diff / count) / 255 * 100;
  }

  _raise(type, message, severity) {
    const alert = {
      id: uid(),
      type,
      message,
      severity,
      time: nowStamp(),
      ts: Date.now(),
    };
    this.alerts.push(alert);
    this.trustScore = Math.max(0, this.trustScore - (this.penalties[type] || 5));
    this.dispatchEvent(new CustomEvent('alert', { detail: alert }));
    this.dispatchEvent(new CustomEvent('score', { detail: this.trustScore }));
  }

  toCSV() {
    const header = 'id,type,severity,time,message\n';
    const rows = this.alerts.map(a =>
      `${a.id},${a.type},${a.severity},${a.time},"${a.message.replace(/"/g, '""')}"`
    ).join('\n');
    return header + rows;
  }

  downloadCSV(filename = 'proctor_log.csv') {
    const blob = new Blob([this.toCSV()], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  verdict() {
    if (this.trustScore >= 80) return 'pass';
    if (this.trustScore >= 50) return 'review';
    return 'fail';
  }
}

/* ---------------------------------- ExamController ---------------------------------- */
/**
 * Manages exam state: countdown timer, question palette, answer tracking.
 */
class ExamController extends EventTarget {
  constructor(questions, opts = {}) {
    super();
    this.questions = questions;              // [{id, text, options: []}]
    this.answers = {};                       // qId -> optionIndex
    this.flagged = new Set();
    this.current = 0;
    this.durationSec = opts.durationSec || 3600;
    this.remaining = this.durationSec;
    this.timer = null;
  }

  start() {
    this.timer = setInterval(() => {
      this.remaining--;
      this.dispatchEvent(new CustomEvent('tick', { detail: this.remaining }));
      if (this.remaining <= 0) this.submit();
    }, 1000);
  }

  stop() { clearInterval(this.timer); }

  answer(qId, optionIndex) {
    this.answers[qId] = optionIndex;
    this.dispatchEvent(new CustomEvent('answered', { detail: { qId, optionIndex } }));
  }

  toggleFlag(qId) {
    this.flagged.has(qId) ? this.flagged.delete(qId) : this.flagged.add(qId);
    this.dispatchEvent(new CustomEvent('flagged', { detail: { qId, flagged: this.flagged.has(qId) } }));
  }

  goto(index) {
    if (index < 0 || index >= this.questions.length) return;
    this.current = index;
    this.dispatchEvent(new CustomEvent('navigate', { detail: index }));
  }

  next() { this.goto(this.current + 1); }
  prev() { this.goto(this.current - 1); }

  statusOf(qId) {
    if (this.flagged.has(qId)) return 'flagged';
    if (this.answers[qId] !== undefined) return 'answered';
    return 'unanswered';
  }

  progress() {
    const answered = Object.keys(this.answers).length;
    return { answered, total: this.questions.length, pct: Math.round((answered / this.questions.length) * 100) };
  }

  formatTime() {
    const h = Math.floor(this.remaining / 3600);
    const m = Math.floor((this.remaining % 3600) / 60);
    const s = this.remaining % 60;
    return h > 0 ? `${pad(h)}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
  }

  submit() {
    this.stop();
    this.dispatchEvent(new CustomEvent('submitted', {
      detail: { answers: this.answers, remaining: this.remaining }
    }));
  }
}

/* ---------------------------------- Admin Dashboard Feed (simulated multi-desk) ---------------------------------- */
/**
 * Drives a simulated multi-student monitoring grid for the admin dashboard.
 * Each desk gets its own trust score drift and randomized alert emission.
 */
class DeskSimulator extends EventTarget {
  constructor(students, opts = {}) {
    super();
    this.desks = students.map(s => ({
      id: s.id,
      name: s.name,
      trustScore: 100,
      status: 'clear',
      alerts: [],
    }));
    this.interval = opts.interval || 4000;
    this.alertTypes = [
      { type: 'no_face', message: 'No face detected', severity: 'high', penalty: 8 },
      { type: 'looking_away', message: 'Looking away from screen', severity: 'medium', penalty: 5 },
      { type: 'multiple_faces', message: 'Multiple faces in frame', severity: 'high', penalty: 12 },
      { type: 'tab_switch', message: 'Tab switch detected', severity: 'medium', penalty: 10 },
    ];
  }

  start() {
    this.timer = setInterval(() => this._tick(), this.interval);
  }

  stop() { clearInterval(this.timer); }

  _tick() {
    this.desks.forEach(desk => {
      if (Math.random() < 0.12) {
        const a = this.alertTypes[Math.floor(Math.random() * this.alertTypes.length)];
        const alert = { ...a, id: uid(), time: nowStamp() };
        desk.alerts.push(alert);
        desk.trustScore = Math.max(0, desk.trustScore - a.penalty);
        desk.status = desk.trustScore >= 80 ? 'clear' : desk.trustScore >= 50 ? 'warning' : 'flagged';
        this.dispatchEvent(new CustomEvent('desk-alert', { detail: { desk, alert } }));
      } else {
        desk.trustScore = Math.min(100, desk.trustScore + 0.3); // slow recovery
      }
    });
    this.dispatchEvent(new CustomEvent('tick', { detail: this.desks }));
  }
}

/* ---------------------------------- Export ---------------------------------- */
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ProctorEngine, ExamController, DeskSimulator, $, $$ };
}