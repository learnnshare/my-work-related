/* =====================================================================
   data.js — real-data loader shim.
   ---------------------------------------------------------------------
   Loads AFTER sim.js. If the pipeline has published window.MI300X_DATA
   (via data/bundle.js), this OVERRIDES the sim.js generators to return
   real captured/normalized records. With no bundle present it is a no-op
   and sim.js remains the offline fallback. Pure file:// friendly — no fetch.
   ===================================================================== */
(function () {
  var DATA = window.MI300X_DATA;
  if (!DATA) {
    console.info('[data.js] no MI300X_DATA bundle — using simulated fallback (sim.js).');
    return;
  }
  console.info('[data.js] MI300X_DATA loaded:', (DATA.records ? Object.keys(DATA.records).length : 0),
               'records,', (DATA.predictions ? Object.keys(DATA.predictions).length : 0), 'predictions.');

  // keep originals as fallback
  var _compute = window.computeMetrics;
  var _predict = window.predictionSet;
  var _gem5 = window.extractGem5Params;
  var _dsp = window.datasetProfile;
  var _curve = window.trainingCurve;
  var _imp = window.featureImportance;
  var _score = window.predictorScorecard;
  var _dev = window.deviceStatus;

  function resolvePrec(cfg) {
    if (cfg.precision) return cfg.precision;
    var w = (typeof WORKLOADS !== 'undefined') && WORKLOADS[cfg.workload];
    return w ? w.pref : 'fp16';
  }
  function key(cfg) {
    return cfg.workload + '|' + resolvePrec(cfg) + '|' + (cfg.batch || 1) + '|' + (cfg.numGPUs || 1);
  }

  // ---- computeMetrics: return the real (device) record when one matches ----
  window.computeMetrics = function (cfg) {
    var rec = DATA.records && DATA.records[key(cfg)];
    if (!rec) return _compute(cfg);
    var wl = (typeof WORKLOADS !== 'undefined' && WORKLOADS[cfg.workload]) || { short: '', name: cfg.workload, cpuBound: 0, regime: '' };
    var m = rec.metrics || {};
    return Object.assign({}, m, {
      cfg: cfg, wl: wl, prec: resolvePrec(cfg),
      layers: rec.layers,
      throughputUnit: m.throughputUnit || (wl.short + '/s'),
      _real: true, _source: rec.meta && rec.meta.source,
    });
  };

  // ---- predictionSet: return the published predicted-vs-measured set ----
  window.predictionSet = function (cfg) {
    var ps = DATA.predictions && DATA.predictions[cfg.workload];
    if (!ps) return _predict(cfg);
    return ps;
  };

  // ---- architect helpers ----
  if (DATA.gem5Params) window.extractGem5Params = function (specKey) { return DATA.gem5Params; };
  if (DATA.datasetProfile) window.datasetProfile = function () { return DATA.datasetProfile; };
  if (DATA.trainReport && DATA.trainReport.curve) {
    window.trainingCurve = function (epochs) {
      var c = DATA.trainReport.curve;
      return { train: c.train, val: c.val, epochs: (c.train || []).length };
    };
  }
  if (DATA.trainReport && DATA.trainReport.featureImportance) {
    window.featureImportance = function () { return DATA.trainReport.featureImportance; };
  }
  if (DATA.trainReport && DATA.trainReport.scorecard) {
    window.predictorScorecard = function () { return DATA.trainReport.scorecard; };
  }
  if (DATA.deviceStatus) {
    window.deviceStatus = function (connected, endpoint) {
      // honor the architect's "use trace instead" toggle; default to real status
      if (connected === false) return _dev(false, endpoint);
      return DATA.deviceStatus;
    };
  }
})();
