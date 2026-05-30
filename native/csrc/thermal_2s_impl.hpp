#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define PY2SESS_HD __host__ __device__
#else
#define PY2SESS_HD
#endif

namespace py2sess_native {

template <typename T>
PY2SESS_HD T* alloc_from(char*& work, std::int64_t count) {
  T* ptr = reinterpret_cast<T*>(work);
  work += sizeof(T) * count;
  return ptr;
}

template <typename T>
PY2SESS_HD T clamp_value(T value, T lower, T upper) {
  return value < lower ? lower : (value > upper ? upper : value);
}

template <typename T>
PY2SESS_HD T exp_cutoff(T value, T cutoff) {
  return value > cutoff ? T(0) : exp(-value);
}

template <typename T>
PY2SESS_HD T exp1_positive(T x) {
  const T euler = T(0.577215664901532860606512090082402431);
  const T eps = T(1.0e-14);
  const T fpmin = T(1.0e-30);
  if (x <= T(0)) {
    return T(1) / T(0);
  }
  if (x > T(1)) {
    T b = x + T(1);
    T c = T(1) / fpmin;
    T d = T(1) / b;
    T h = d;
    for (int i = 1; i <= 100; ++i) {
      const T a = -static_cast<T>(i * i);
      b += T(2);
      d = T(1) / (a * d + b);
      c = b + a / c;
      const T del = c * d;
      h *= del;
      if (fabs(del - T(1)) <= eps) {
        break;
      }
    }
    return h * exp(-x);
  }

  T ans = -log(x) - euler;
  T fact = T(1);
  for (int i = 1; i <= 100; ++i) {
    fact *= -x / static_cast<T>(i);
    const T del = -fact / static_cast<T>(i);
    ans += del;
    if (fabs(del) < fabs(ans) * eps) {
      break;
    }
  }
  return ans;
}

template <typename T>
PY2SESS_HD T expn2_positive(T x) {
  if (x <= T(0)) {
    return T(1);
  }
  if (x > T(88)) {
    return T(0);
  }
  return exp(-x) - x * exp1_positive(x);
}

template <typename T>
PY2SESS_HD T expn3_positive(T x) {
  if (x <= T(0)) {
    return T(0.5);
  }
  if (x > T(88)) {
    return T(0);
  }
  const T e2 = expn2_positive(x);
  return T(0.5) * (exp(-x) - x * e2);
}

template <typename T>
PY2SESS_HD T taylor_series_1(int order, T eps, T delta, T udel, T sm) {
  const int mterms = order + 1;
  T dm1 = delta;
  T mult = dm1;
  T power = T(1);
  for (int m = 2; m <= mterms; ++m) {
    dm1 = delta * dm1 / static_cast<T>(m);
    power = power * eps;
    mult = mult + power * dm1;
  }
  return mult * udel * sm;
}

template <typename T>
PY2SESS_HD T taylor_series_2(
    int order,
    T small,
    T eps,
    T y,
    T delta,
    T fac1,
    T fac2,
    T sm) {
  const int max_terms = 10;
  int mterms = order + 1;
  if (fabs(y) < small) {
    mterms += 1;
  }

  T d[max_terms + 1] = {};
  T dm1 = T(1);
  d[0] = dm1;
  for (int m = 1; m <= mterms; ++m) {
    dm1 = delta * dm1 / static_cast<T>(m);
    d[m] = dm1;
  }

  if (fabs(y) < small) {
    T power = T(1);
    T power2 = T(1);
    T mult = d[2];
    for (int m = 3; m <= mterms; ++m) {
      power = power * (eps - y);
      power2 = power - y * power2;
      mult = mult + d[m] * power2;
    }
    return mult * fac1 * sm;
  }

  const T y1 = T(1) / y;
  T ac[max_terms + 1] = {};
  T acm1 = T(1);
  ac[0] = acm1;
  for (int m = 1; m <= mterms; ++m) {
    acm1 = y1 * acm1;
    ac[m] = acm1;
  }

  T cc[max_terms + 1] = {};
  cc[0] = T(1);
  for (int m = 1; m <= mterms; ++m) {
    T total = T(0);
    for (int j = 0; j <= m; ++j) {
      total += ac[j] * d[m - j];
    }
    cc[m] = total;
  }

  T power = T(1);
  T mult = fac1 * ac[1] - fac2 * cc[1];
  for (int m = 2; m <= mterms; ++m) {
    power = eps * power;
    mult = mult + power * (fac1 * ac[m] - fac2 * cc[m]);
  }
  return mult * sm * y1;
}

template <typename T>
PY2SESS_HD std::size_t thermal_2s_workspace_bytes(std::int64_t nlay) {
  const auto layers = nlay > 0 ? static_cast<std::size_t>(nlay) : 1;
  const auto ntotal = 2 * layers;
  const auto layer_arrays = 15 * layers;
  const auto bvp_arrays = ntotal + (ntotal > 1 ? ntotal - 1 : 1) +
                          (ntotal > 2 ? ntotal - 2 : 1);
  return (layer_arrays + bvp_arrays + 8) * sizeof(T);
}

template <typename T>
PY2SESS_HD std::size_t solar_2s_workspace_bytes(std::int64_t nlay) {
  const auto layers = nlay > 0 ? static_cast<std::size_t>(nlay) : 1;
  const auto ntotal = 2 * layers;
  const auto layer_arrays = 24 * layers;
  const auto bvp_arrays = ntotal + (ntotal > 1 ? ntotal - 1 : 1) +
                          (ntotal > 2 ? ntotal - 2 : 1);
  return (layer_arrays + bvp_arrays + 8) * sizeof(T);
}

template <typename T>
PY2SESS_HD std::size_t solar_2s_flux_workspace_bytes(std::int64_t nlay) {
  const auto layers = nlay > 0 ? static_cast<std::size_t>(nlay) : 1;
  const auto ntotal = 2 * layers;
  const auto layer_arrays = 11 * layers;
  const auto bvp_arrays = ntotal + (ntotal > 1 ? ntotal - 1 : 1) +
                          (ntotal > 2 ? ntotal - 2 : 1);
  return (layer_arrays + bvp_arrays + 8) * sizeof(T);
}

template <typename T>
PY2SESS_HD std::size_t solar_fo_workspace_bytes(std::int64_t nlay, bool return_profile) {
  const auto layers = nlay > 0 ? static_cast<std::size_t>(nlay) : 1;
  const auto profile_arrays = return_profile ? 2 * layers : 0;
  return (layers + profile_arrays + 8) * sizeof(T);
}

template <typename T>
PY2SESS_HD std::size_t solar_fo_pp_workspace_bytes(std::int64_t nlay) {
  const auto layers = nlay > 0 ? static_cast<std::size_t>(nlay) : 1;
  const auto levels = layers + 1;
  return (4 * layers + levels + 8) * sizeof(T);
}

template <typename T>
PY2SESS_HD std::size_t thermal_fo_workspace_bytes(std::int64_t nlay) {
  const auto layers = nlay > 0 ? static_cast<std::size_t>(nlay) : 1;
  return (2 * layers + 8) * sizeof(T);
}

template <typename T>
PY2SESS_HD std::size_t thermal_fo_flux_workspace_bytes(std::int64_t nlay) {
  const auto layers = nlay > 0 ? static_cast<std::size_t>(nlay) : 1;
  const auto levels = layers + 1;
  return (3 * layers + 2 * levels + 8) * sizeof(T);
}

template <typename T>
PY2SESS_HD void gauss_legendre_8_positive(int index, T* mu, T* weight) {
  const T mu_values[8] = {
      T(0.019855071751231884),
      T(0.10166676129318664),
      T(0.2372337950418355),
      T(0.4082826787521751),
      T(0.5917173212478249),
      T(0.7627662049581645),
      T(0.8983332387068134),
      T(0.9801449282487681),
  };
  const T weight_values[8] = {
      T(0.05061426814518813),
      T(0.11119051722668724),
      T(0.15685332293894363),
      T(0.18134189168918099),
      T(0.18134189168918099),
      T(0.15685332293894363),
      T(0.11119051722668724),
      T(0.05061426814518813),
  };
  *mu = mu_values[index];
  *weight = weight_values[index];
}

template <typename T>
PY2SESS_HD void thermal_fo_profiles_for_mu(
    int nlay,
    const T* deltaus,
    const T* therm0,
    const T* therm1,
    T mu,
    T surface_source,
    T* up,
    T* down) {
  T cumulative = surface_source;
  up[nlay] = cumulative;
  for (int n = nlay - 1; n >= 0; --n) {
    const T lostau = deltaus[n] / mu;
    if (lostau <= T(1.0e-10)) {
      up[n] = cumulative;
      continue;
    }
    T trans = T(0);
    T one_minus_trans = T(1);
    if (lostau <= T(88)) {
      trans = exp(-lostau);
      if (fabs(lostau) < T(1.0e-4)) {
        const T x = lostau;
        one_minus_trans =
            x * (T(1) - x * (T(0.5) - x * (T(1.0 / 6.0) - x * T(1.0 / 24.0))));
      } else {
        one_minus_trans = T(1) - trans;
      }
    }
    const T ratio = lostau == T(0) ? T(1) : one_minus_trans / lostau;
    const T source_delta = therm1[n] * deltaus[n];
    const T source = therm0[n] * one_minus_trans + source_delta * (ratio - trans);
    cumulative = trans * cumulative + source;
    up[n] = cumulative;
  }

  cumulative = T(0);
  down[0] = cumulative;
  for (int n = 0; n < nlay; ++n) {
    const T lostau = deltaus[n] / mu;
    if (lostau <= T(1.0e-10)) {
      down[n + 1] = cumulative;
      continue;
    }
    T trans = T(0);
    T one_minus_trans = T(1);
    if (lostau <= T(88)) {
      trans = exp(-lostau);
      if (fabs(lostau) < T(1.0e-4)) {
        const T x = lostau;
        one_minus_trans =
            x * (T(1) - x * (T(0.5) - x * (T(1.0 / 6.0) - x * T(1.0 / 24.0))));
      } else {
        one_minus_trans = T(1) - trans;
      }
    }
    const T ratio = lostau == T(0) ? T(1) : one_minus_trans / lostau;
    const T source_delta = therm1[n] * deltaus[n];
    const T source = therm0[n] * one_minus_trans + source_delta * (T(1) - ratio);
    cumulative = source + trans * cumulative;
    down[n + 1] = cumulative;
  }
}

template <typename T>
PY2SESS_HD void thermal_fo_flux_correction_row(
    int nlay,
    T stream_value,
    bool do_optical_deltam_scaling,
    bool do_source_deltam_scaling,
    int n_mu,
    const T* mu_nodes,
    const T* mu_weights,
    const T* tau,
    const T* omega,
    const T* scaling,
    const T* planck,
    const T* surfbb,
    const T* emissivity,
    T* out,
    char* work) {
  const int nlev = nlay + 1;
  const T pi = T(3.141592653589793238462643383279502884);
  const T pi2 = T(2) * pi;
  T* flux_up = out;
  T* flux_down = flux_up + nlev;
  T* flux_net = flux_down + nlev;
  T* flux_mean = flux_net + nlev;
  T* deltaus = alloc_from<T>(work, nlay);
  T* therm0 = alloc_from<T>(work, nlay);
  T* therm1 = alloc_from<T>(work, nlay);
  T* up = alloc_from<T>(work, nlev);
  T* down = alloc_from<T>(work, nlev);

  for (int level = 0; level < nlev; ++level) {
    flux_up[level] = T(0);
    flux_down[level] = T(0);
    flux_net[level] = T(0);
    flux_mean[level] = T(0);
  }

  for (int n = 0; n < nlay; ++n) {
    const T omfac = T(1) - omega[n] * scaling[n];
    T delta = do_optical_deltam_scaling ? tau[n] * omfac : tau[n];
    if (delta <= T(0)) {
      delta = T(1.0e-12);
    }
    deltaus[n] = delta;
    T source_scale = T(1) - omega[n];
    if (do_source_deltam_scaling) {
      source_scale = source_scale / omfac;
    }
    therm0[n] = planck[n] * source_scale;
    therm1[n] = (planck[n + 1] - planck[n]) * source_scale / delta;
  }

  const T surface_source = (*surfbb) * (*emissivity);
  for (int q = 0; q < n_mu; ++q) {
    const T mu = mu_nodes[q];
    const T weight = mu_weights[q];
    thermal_fo_profiles_for_mu(nlay, deltaus, therm0, therm1, mu, surface_source, up, down);
    for (int level = 0; level < nlev; ++level) {
      flux_up[level] += pi2 * weight * mu * up[level];
      flux_down[level] += pi2 * weight * mu * down[level];
      flux_mean[level] += T(0.5) * weight * (up[level] + down[level]);
    }
  }

  thermal_fo_profiles_for_mu(
      nlay, deltaus, therm0, therm1, stream_value, surface_source, up, down);
  for (int level = 0; level < nlev; ++level) {
    flux_up[level] -= pi2 * stream_value * up[level];
    flux_down[level] -= pi2 * stream_value * down[level];
    flux_mean[level] -= T(0.5) * (up[level] + down[level]);
    flux_net[level] = flux_up[level] - flux_down[level];
  }
  for (int level = nlay - 1; level >= 0; --level) {
    const T omfac = T(1) - omega[level] * scaling[level];
    const T delta = do_optical_deltam_scaling ? tau[level] * omfac : tau[level];
    if (delta <= T(0)) {
      flux_up[level] = flux_up[level + 1];
      flux_down[level] = flux_down[level + 1];
      flux_mean[level] = flux_mean[level + 1];
      flux_net[level] = flux_up[level] - flux_down[level];
    }
  }
}

template <typename T>
PY2SESS_HD void thermal_fo_row(
    int nlay,
    int nfine,
    bool do_nadir,
    bool do_optical_deltam_scaling,
    bool do_source_deltam_scaling,
    const T* tau,
    const T* omega,
    const T* scaling,
    const T* planck,
    const T* surfbb,
    const T* emissivity,
    const T* height_delta,
    const T* xfine,
    const T* wfine,
    const T* cota,
    const T* cotfine,
    const T* csqfine,
    T rayconv,
    bool return_components,
    bool return_profile,
    T* out,
    char* work) {
  const int nlev = nlay + 1;
  T* layer_source = alloc_from<T>(work, nlay);
  T* layer_trans = alloc_from<T>(work, nlay);
  T trans_to_top = T(1);
  T cum_atmos = T(0);

  for (int layer = 0; layer < nlay; ++layer) {
    const T optical_factor =
        do_optical_deltam_scaling ? (T(1) - omega[layer] * scaling[layer]) : T(1);
    const T deltaus = tau[layer] * optical_factor;
    T source_scale = T(1) - omega[layer];
    if (do_source_deltam_scaling) {
      source_scale = source_scale / (T(1) - omega[layer] * scaling[layer]);
    }
    const T therm0 = planck[layer] * source_scale;
    const T therm1 = deltaus == T(0)
        ? T(0)
        : ((planck[layer + 1] - planck[layer]) / deltaus) * source_scale;
    const T extinction = deltaus / height_delta[layer];

    T source = T(0);
    T lostrans = T(0);
    if (do_nadir) {
      lostrans = exp_cutoff(deltaus, T(88));
      for (int j = 0; j < nfine; ++j) {
        const int index = j * nlay + layer;
        const T xjkn = xfine[index] * extinction;
        const T solution = therm0 + xjkn * therm1;
        source += solution * extinction * exp(-xjkn) * wfine[index];
      }
    } else {
      const T cot_upper = cota[layer];
      const T cot_lower = cota[layer + 1];
      const T ke = rayconv * extinction;
      const T lostau = ke * (cot_upper - cot_lower);
      lostrans = exp_cutoff(lostau, T(88));
      for (int j = 0; j < nfine; ++j) {
        const int index = j * nlay + layer;
        const T xjkn = xfine[index] * extinction;
        const T solution = therm0 + xjkn * therm1;
        const T optical_path = ke * (cot_upper - cotfine[index]);
        source += solution * ke * csqfine[index] * wfine[index] * exp(-optical_path);
      }
    }

    cum_atmos += trans_to_top * source;
    trans_to_top *= lostrans;
    layer_source[layer] = source;
    layer_trans[layer] = lostrans;
  }

  const T surface = trans_to_top * (*surfbb) * (*emissivity);
  if (return_profile) {
    T* total_profile = out;
    T* atmos_profile = return_components ? total_profile + nlev : nullptr;
    T* surface_profile = return_components ? atmos_profile + nlev : nullptr;
    T atmos_accum = T(0);
    T surface_accum = (*surfbb) * (*emissivity);
    if (return_components) {
      atmos_profile[nlay] = T(0);
      surface_profile[nlay] = surface_accum;
    }
    total_profile[nlay] = surface_accum;
    for (int layer = nlay - 1; layer >= 0; --layer) {
      atmos_accum = layer_source[layer] + layer_trans[layer] * atmos_accum;
      surface_accum = layer_trans[layer] * surface_accum;
      total_profile[layer] = atmos_accum + surface_accum;
      if (return_components) {
        atmos_profile[layer] = atmos_accum;
        surface_profile[layer] = surface_accum;
      }
    }
    return;
  }

  out[0] = cum_atmos + surface;
  if (return_components) {
    out[1] = cum_atmos;
    out[2] = surface;
  }
}

template <typename T>
PY2SESS_HD void solar_fo_row(
    int nlay,
    int nfine,
    int ntrav_nl,
    int ntrav_max,
    bool do_nadir,
    T mu0,
    T rayconv,
    const T* tau,
    const T* omega,
    const T* scaling,
    const T* albedo,
    const T* flux_factor,
    const T* exact_scatter,
    const T* inv_layer_thickness,
    const T* sunpathsnl,
    const T* cota,
    const T* cotfine,
    const T* csqfine,
    const T* wfine,
    const T* xfine,
    const T* sunpathsfine,
    const std::int64_t* nfinedivs,
    const std::int64_t* ntraversefine,
    T* out,
    bool return_components,
    bool return_profile,
    char* work) {
  T* extinction = alloc_from<T>(work, nlay);
  T* profile_up = return_profile ? alloc_from<T>(work, nlay) : nullptr;
  T* profile_db = return_profile ? alloc_from<T>(work, nlay) : nullptr;
  for (int layer = 0; layer < nlay; ++layer) {
    const T tau_scaled = tau[layer] * (T(1) - omega[layer] * scaling[layer]);
    extinction[layer] = tau_scaled * inv_layer_thickness[layer];
  }

  T total_tau = T(0);
  for (int k = 0; k < ntrav_nl; ++k) {
    total_tau += extinction[k] * sunpathsnl[k];
  }
  const T attenuation_nl = exp_cutoff(total_tau, T(88));
  T cumsource_up = T(0);
  T cumsource_db = T(4) * mu0 * (*albedo) * attenuation_nl;
  const T surface_db = cumsource_db;

  for (int n = nlay; n > 0; --n) {
    const int layer = n - 1;
    T layer_sum = T(0);
    const int nfine_layer = static_cast<int>(nfinedivs[layer]);
    if (do_nadir) {
      const T tau_scaled_layer = tau[layer] * (T(1) - omega[layer] * scaling[layer]);
      const T kn = extinction[layer];
      for (int j = 0; j < nfine_layer; ++j) {
        const int ntrav = static_cast<int>(ntraversefine[j * nlay + layer]);
        T fine_tau = T(0);
        for (int k = 0; k < ntrav; ++k) {
          fine_tau += extinction[k] * sunpathsfine[(k * nfine + j) * nlay + layer];
        }
        const T attenuation = exp_cutoff(fine_tau, T(88));
        const int index = j * nlay + layer;
        layer_sum += exact_scatter[layer] * attenuation * exp(-xfine[index] * kn) * wfine[index];
      }
      const T lostrans = exp_cutoff(tau_scaled_layer, T(88));
      const T source = layer_sum * kn;
      cumsource_db = lostrans * cumsource_db;
      cumsource_up = lostrans * cumsource_up + source;
    } else {
      const T cot_2 = cota[layer];
      const T cot_1 = cota[layer + 1];
      const T ke = rayconv * extinction[layer];
      const T lostrans = exp(-ke * (cot_2 - cot_1));
      for (int j = 0; j < nfine_layer; ++j) {
        const int ntrav = static_cast<int>(ntraversefine[j * nlay + layer]);
        T fine_tau = T(0);
        for (int k = 0; k < ntrav; ++k) {
          fine_tau += extinction[k] * sunpathsfine[(k * nfine + j) * nlay + layer];
        }
        const T attenuation = exp_cutoff(fine_tau, T(88));
        const int index = j * nlay + layer;
        const T tran = exp(-ke * (cot_2 - cotfine[index]));
        layer_sum +=
            exact_scatter[layer] * attenuation * csqfine[index] * tran * wfine[index];
      }
      const T source = layer_sum * ke;
      cumsource_db = lostrans * cumsource_db;
      cumsource_up = lostrans * cumsource_up + source;
    }
    if (return_profile) {
      profile_up[layer] = cumsource_up;
      profile_db[layer] = cumsource_db;
    }
  }

  const T pi = T(3.141592653589793238462643383279502884);
  const T scale = T(0.25) * (*flux_factor) / pi;
  if (return_profile) {
    const int nlev = nlay + 1;
    T* total_profile = out;
    T* ss_profile = return_components ? total_profile + nlev : nullptr;
    T* db_profile = return_components ? ss_profile + nlev : nullptr;
    for (int level = 0; level < nlay; ++level) {
      const T single_scatter_level = scale * profile_up[level];
      const T direct_beam_level = scale * profile_db[level];
      total_profile[level] = single_scatter_level + direct_beam_level;
      if (return_components) {
        ss_profile[level] = single_scatter_level;
        db_profile[level] = direct_beam_level;
      }
    }
    total_profile[nlay] = scale * surface_db;
    if (return_components) {
      ss_profile[nlay] = T(0);
      db_profile[nlay] = scale * surface_db;
    }
    return;
  }

  const T single_scatter = scale * cumsource_up;
  const T direct_beam = scale * cumsource_db;
  out[0] = single_scatter + direct_beam;
  if (return_components) {
    out[1] = single_scatter;
    out[2] = direct_beam;
  }
}

template <typename T>
PY2SESS_HD void solar_fo_pp_row(
    int nlay,
    T mu0,
    T user_stream,
    const T* tau,
    const T* omega,
    const T* scaling,
    const T* surface_reflectance,
    const T* flux_factor,
    const T* exact_scatter,
    T* out,
    bool return_profile,
    char* work) {
  const int nlev = nlay + 1;
  T* delta = alloc_from<T>(work, nlay);
  T* attenuation = alloc_from<T>(work, nlev);
  T* lostrans = alloc_from<T>(work, nlay);
  T* sources = alloc_from<T>(work, nlay);
  T* solutions = alloc_from<T>(work, nlay);

  T cumulative_tau = T(0);
  attenuation[0] = T(1);
  for (int n = 0; n < nlay; ++n) {
    delta[n] = tau[n] * (T(1) - omega[n] * scaling[n]);
    cumulative_tau += delta[n];
    attenuation[n + 1] = exp_cutoff(cumulative_tau / mu0, T(88));
  }

  T previous_attenuation = attenuation[0];
  if (fabs(user_stream) <= T(1.0e-12)) {
    for (int n = 0; n < nlay; ++n) {
      solutions[n] = exact_scatter[n] * previous_attenuation;
      sources[n] = solutions[n];
      lostrans[n] = T(0);
      previous_attenuation = attenuation[n + 1];
    }
  } else {
    const T factor2 = user_stream / mu0;
    for (int n = 0; n < nlay; ++n) {
      const T trans = exp_cutoff(delta[n] / user_stream, T(88));
      const T current_attenuation = attenuation[n + 1];
      const T factor1 =
          previous_attenuation == T(0) ? T(0) : current_attenuation / previous_attenuation;
      solutions[n] = exact_scatter[n] * previous_attenuation;
      sources[n] = solutions[n] * (T(1) - factor1 * trans) / (factor2 + T(1));
      lostrans[n] = trans;
      previous_attenuation = current_attenuation;
    }
  }

  const T pi = T(3.141592653589793238462643383279502884);
  const T scale = T(0.25) * (*flux_factor) / pi;
  T cumsource_up = T(0);
  T cumsource_db = T(4) * mu0 * (*surface_reflectance) * attenuation[nlay];
  const T surface_db = cumsource_db;

  if (return_profile) {
    out[nlay] = scale * surface_db;
    for (int n = nlay - 1; n >= 0; --n) {
      cumsource_db = lostrans[n] * cumsource_db;
      cumsource_up = lostrans[n] * cumsource_up + sources[n];
      out[n] = scale * (cumsource_up + cumsource_db);
    }
    return;
  }

  for (int n = nlay - 1; n >= 0; --n) {
    cumsource_db = lostrans[n] * cumsource_db;
    cumsource_up = lostrans[n] * cumsource_up + sources[n];
  }
  out[0] = scale * (cumsource_up + cumsource_db);
}

template <typename T>
PY2SESS_HD void solar_fo_flux_correction_row(
    int nlay,
    T stream_value,
    T mu0,
    bool do_optical_deltam_scaling,
    const T* tau,
    const T* omega,
    const T* scaling,
    const T* surface_reflectance,
    const T* flux_factor,
    T* out) {
  const int nlev = nlay + 1;
  const T pi = T(3.141592653589793238462643383279502884);
  const T surface = *surface_reflectance;
  T exact_total = T(0);
  T embedded_total = T(0);
  for (int n = 0; n < nlay; ++n) {
    const T twostream_delta = tau[n] * (T(1) - omega[n] * scaling[n]);
    embedded_total += twostream_delta;
    exact_total += do_optical_deltam_scaling ? twostream_delta : tau[n];
  }

  const T exact_surface_flux =
      (*flux_factor) * mu0 * surface * exp_cutoff(exact_total / mu0, T(88));
  const T embedded_surface_flux =
      (*flux_factor) * mu0 * surface * exp_cutoff(embedded_total / mu0, T(88));

  T exact_distance = exact_total;
  T embedded_distance = embedded_total;
  T* flux_up = out;
  T* flux_down = flux_up + nlev;
  T* flux_net = flux_down + nlev;
  T* flux_mean = flux_net + nlev;
  for (int level = 0; level < nlev; ++level) {
    const T exact_up = T(2) * exact_surface_flux * expn3_positive(exact_distance);
    const T exact_mean =
        T(0.5) * exact_surface_flux * expn2_positive(exact_distance) / pi;
    const T embedded_trans = exp_cutoff(embedded_distance / stream_value, T(88));
    const T embedded_up = T(2) * stream_value * embedded_surface_flux * embedded_trans;
    const T embedded_mean = embedded_surface_flux * embedded_trans / (T(2) * pi);

    flux_up[level] = exact_up - embedded_up;
    flux_down[level] = T(0);
    flux_net[level] = flux_up[level];
    flux_mean[level] = exact_mean - embedded_mean;

    if (level < nlay) {
      const T twostream_delta = tau[level] * (T(1) - omega[level] * scaling[level]);
      exact_distance -= do_optical_deltam_scaling ? twostream_delta : tau[level];
      embedded_distance -= twostream_delta;
    }
  }
}

template <typename T>
PY2SESS_HD void solve_bvp_row(
    int nlay,
    T albedo,
    T bottom_source,
    T surface_factor,
    T stream_value,
    const T* xpos1,
    const T* xpos2,
    const T* eigentrans,
    const T* wupper0,
    const T* wupper1,
    const T* wlower0,
    const T* wlower1,
    T* lcon,
    T* mcon,
    T* col,
    T* elm1,
    T* elm2) {
  const int ntotal = 2 * nlay;
  const T factor = surface_factor * albedo;
  const T xpnet = xpos2[nlay - 1] - factor * xpos1[nlay - 1] * stream_value;
  const T xmnet = xpos1[nlay - 1] - factor * xpos2[nlay - 1] * stream_value;

  col[0] = -wupper0[0];
  for (int n = 1; n < nlay; ++n) {
    const int prev = n - 1;
    const int row_m = 2 * n - 1;
    const int row_p = row_m + 1;
    col[row_m] = wupper0[n] - wlower0[prev];
    col[row_p] = wupper1[n] - wlower1[prev];
  }
  col[ntotal - 1] =
      -wlower1[nlay - 1] + wlower0[nlay - 1] * stream_value * factor + bottom_source;

  if (nlay == 1) {
    const T a00 = xpos1[0];
    const T a01 = xpos2[0] * eigentrans[0];
    const T a10 = xpnet * eigentrans[0];
    const T a11 = xmnet;
    const T rhs0 = col[0];
    const T rhs1 = col[1];
    const T det = a00 * a11 - a01 * a10;
    lcon[0] = (rhs0 * a11 - a01 * rhs1) / det;
    mcon[0] = (a00 * rhs1 - rhs0 * a10) / det;
    return;
  }

  T elm31 = T(1) / xpos1[0];
  T elm1_i2 = -(xpos2[0] * eigentrans[0]) * elm31;
  elm1[0] = elm1_i2;
  T elm2_i2 = T(0);
  elm2[0] = elm2_i2;
  T col_i2 = col[0] * elm31;
  col[0] = col_i2;

  T mat22 = xpos1[0] * eigentrans[0];
  T bet = xpos2[0] + mat22 * elm1_i2;
  bet = -T(1) / bet;
  T elm1_i1 = -xpos1[1] * bet;
  elm1[1] = elm1_i1;
  T elm2_i1 = (-xpos2[1] * eigentrans[1]) * bet;
  elm2[1] = elm2_i1;
  T col_i1 = (mat22 * col_i2 - col[1]) * bet;
  col[1] = col_i1;

  for (int i = 2; i < ntotal - 2; ++i) {
    T mat1_i;
    T mat2_i;
    T mat3_i;
    T mat4_i;
    T mat5_i;
    if (i % 2 == 0) {
      const int n = i / 2;
      const int prev = n - 1;
      mat1_i = xpos2[prev] * eigentrans[prev];
      mat2_i = xpos1[prev];
      mat3_i = -xpos2[n];
      mat4_i = -xpos1[n] * eigentrans[n];
      mat5_i = T(0);
    } else {
      const int n = (i + 1) / 2;
      const int prev = n - 1;
      mat1_i = T(0);
      mat2_i = xpos1[prev] * eigentrans[prev];
      mat3_i = xpos2[prev];
      mat4_i = -xpos1[n];
      mat5_i = -xpos2[n] * eigentrans[n];
    }
    bet = mat2_i + mat1_i * elm1_i2;
    T den = mat3_i + mat1_i * elm2_i2 + bet * elm1_i1;
    den = -T(1) / den;
    elm1_i2 = elm1_i1;
    elm1_i1 = (mat4_i + bet * elm2_i1) * den;
    elm1[i] = elm1_i1;
    elm2_i2 = elm2_i1;
    elm2_i1 = mat5_i * den;
    elm2[i] = elm2_i1;
    const T col_i = (mat1_i * col_i2 + bet * col_i1 - col[i]) * den;
    col_i2 = col_i1;
    col_i1 = col_i;
    col[i] = col_i;
  }

  int i = ntotal - 2;
  const int n = i / 2;
  const int prev = n - 1;
  T mat1_i = xpos2[prev] * eigentrans[prev];
  T mat2_i = xpos1[prev];
  T mat3_i = -xpos2[n];
  T mat4_i = -xpos1[n] * eigentrans[n];
  bet = mat2_i + mat1_i * elm1_i2;
  T den = mat3_i + mat1_i * elm2_i2 + bet * elm1_i1;
  den = -T(1) / den;
  elm1_i2 = elm1_i1;
  elm1_i1 = (mat4_i + bet * elm2_i1) * den;
  elm1[i] = elm1_i1;
  elm2_i2 = elm2_i1;
  T col_i = (mat1_i * col_i2 + bet * col_i1 - col[i]) * den;
  col_i2 = col_i1;
  col_i1 = col_i;
  col[i] = col_i;

  i = ntotal - 1;
  bet = xpnet * eigentrans[nlay - 1];
  den = xmnet + bet * elm1_i1;
  den = -T(1) / den;
  col_i = (bet * col_i1 - col[i]) * den;
  col_i2 = col_i1;
  col_i1 = col_i;
  col[i] = col_i;

  i = ntotal - 2;
  col_i = col_i2 + elm1[i] * col_i1;
  col[i] = col_i;
  col_i2 = col_i1;
  col_i1 = col_i;
  for (i = ntotal - 3; i >= 0; --i) {
    col_i = col[i] + elm1[i] * col_i1 + elm2[i] * col_i2;
    col[i] = col_i;
    col_i2 = col_i1;
    col_i1 = col_i;
  }

  for (int layer = 0; layer < nlay; ++layer) {
    lcon[layer] = col[2 * layer];
    mcon[layer] = col[2 * layer + 1];
  }
}

template <typename T>
PY2SESS_HD void solve_thermal_bvp_row(
    int nlay,
    T albedo,
    T emissivity,
    T surfbb,
    T surface_factor,
    T stream_value,
    const T* xpos1,
    const T* xpos2,
    const T* eigentrans,
    const T* wupper0,
    const T* wupper1,
    const T* wlower0,
    const T* wlower1,
    T* lcon,
    T* mcon,
    T* col,
    T* elm1,
    T* elm2) {
  solve_bvp_row(
      nlay,
      albedo,
      surfbb * emissivity,
      surface_factor,
      stream_value,
      xpos1,
      xpos2,
      eigentrans,
      wupper0,
      wupper1,
      wlower0,
      wlower1,
      lcon,
      mcon,
      col,
      elm1,
      elm2);
}

template <typename T>
PY2SESS_HD void thermal_2s_row(
    int nlay,
    T stream_value,
    T user_stream,
    T thermal_tcutoff,
    const T* tau,
    const T* omega,
    const T* asymm,
    const T* scaling,
    const T* planck,
    const T* surfbb,
    const T* emissivity,
    const T* albedo,
    const T* brdf_f,
    const T* ubrdf_f,
    T* out,
    bool return_profile,
    bool return_fluxes,
    bool do_upwelling,
    bool do_dnwelling,
    bool use_brdf,
    char* work) {
  const int nlev = nlay + 1;
  const int radiance_count = return_profile ? nlev : 1;
  T* flux_up = return_fluxes ? out + radiance_count : nullptr;
  T* flux_down = return_fluxes ? flux_up + nlev : nullptr;
  T* flux_net = return_fluxes ? flux_down + nlev : nullptr;
  T* flux_mean = return_fluxes ? flux_net + nlev : nullptr;
  T* eigentrans = alloc_from<T>(work, nlay);
  T* xpos1 = alloc_from<T>(work, nlay);
  T* xpos2 = alloc_from<T>(work, nlay);
  T* t_delt_userm = alloc_from<T>(work, nlay);
  T* u_xpos = alloc_from<T>(work, nlay);
  T* u_xneg = alloc_from<T>(work, nlay);
  T* hmult_1 = alloc_from<T>(work, nlay);
  T* hmult_2 = alloc_from<T>(work, nlay);
  T* t_wupper0 = alloc_from<T>(work, nlay);
  T* t_wupper1 = alloc_from<T>(work, nlay);
  T* t_wlower0 = alloc_from<T>(work, nlay);
  T* t_wlower1 = alloc_from<T>(work, nlay);
  T* layer_tsup_up = alloc_from<T>(work, nlay);
  T* lcon = alloc_from<T>(work, nlay);
  T* mcon = alloc_from<T>(work, nlay);
  T* col = alloc_from<T>(work, 2 * nlay);
  T* elm1 = alloc_from<T>(work, 2 * nlay > 1 ? 2 * nlay - 1 : 1);
  T* elm2 = alloc_from<T>(work, 2 * nlay > 2 ? 2 * nlay - 2 : 1);

  const T xinv = T(1) / stream_value;
  const T pxsq = stream_value * stream_value;
  const T user_secant = T(1) / user_stream;
  const T hmu_stream = T(0.5) * stream_value;

  bool transparent = true;
  for (int i = 0; i < nlay; ++i) {
    const T omfac = T(1) - omega[i] * scaling[i];
    if (tau[i] * omfac != T(0)) {
      transparent = false;
      break;
    }
  }

  if (transparent) {
    const int out_count = return_profile ? nlay + 1 : 1;
    for (int i = 0; i < out_count; ++i) {
      out[i] = T(0);
    }
    if (return_fluxes) {
      const T pi = T(3.141592653589793238462643383279502884);
      const T pi2 = T(2) * pi;
      const T surface_emission = (*surfbb) * (*emissivity);
      const T up_value = do_upwelling ? pi2 * stream_value * surface_emission : T(0);
      const T mean_value = do_upwelling ? T(0.5) * surface_emission : T(0);
      for (int i = 0; i < nlev; ++i) {
        flux_up[i] = up_value;
        flux_down[i] = T(0);
        flux_net[i] = up_value;
        flux_mean[i] = mean_value;
      }
    }
    return;
  }

  for (int i = 0; i < nlay; ++i) {
    const T omfac = T(1) - omega[i] * scaling[i];
    const T m1fac = T(1) - scaling[i];
    const T delta_tau = tau[i] * omfac;
    const T omega_total = clamp_value((m1fac * omega[i]) / omfac, T(1.0e-9), T(0.999999999));
    T asymm_total = clamp_value((asymm[i] - scaling[i]) / m1fac, T(-0.999999999), T(0.999999999));
    if (asymm_total >= T(0) && asymm_total < T(1.0e-9)) {
      asymm_total = T(1.0e-9);
    } else if (asymm_total < T(0) && asymm_total > T(-1.0e-9)) {
      asymm_total = T(-1.0e-9);
    }
    const T therm0 = planck[i];
    const T therm1 = delta_tau == T(0) ? T(0) : (planck[i + 1] - planck[i]) / delta_tau;

    const T omega_asymm_3 = T(3) * omega_total * asymm_total;
    const T sab = xinv * (omega_total - T(1));
    const T dab = xinv * (pxsq * omega_asymm_3 - T(1));
    const T eigenvalue = sqrt(sab * dab);
    const T eigentrans_i = exp_cutoff(eigenvalue * delta_tau, T(88));
    eigentrans[i] = eigentrans_i;
    const T difvec = -sab / eigenvalue;
    const T xpos1_i = T(0.5) * (T(1) + difvec);
    const T xpos2_i = T(0.5) * (T(1) - difvec);
    xpos1[i] = xpos1_i;
    xpos2[i] = xpos2_i;
    const T norm_saved = stream_value * (xpos1_i * xpos1_i - xpos2_i * xpos2_i);

    const T t_delt_userm_i = exp(-delta_tau * user_secant);
    t_delt_userm[i] = t_delt_userm_i;
    const T u_help_p0 = (xpos2_i + xpos1_i) * T(0.5);
    const T u_help_p1 = (xpos2_i - xpos1_i) * hmu_stream;
    const T u_xpos_i = u_help_p0 * omega_total + u_help_p1 * omega_asymm_3 * user_stream;
    const T u_xneg_i = u_help_p0 * omega_total - u_help_p1 * omega_asymm_3 * user_stream;
    u_xpos[i] = u_xpos_i;
    u_xneg[i] = u_xneg_i;

    const T zp = user_secant + eigenvalue;
    const T zm = user_secant - eigenvalue;
    const T hmult_2_i = user_secant * (T(1) - eigentrans_i * t_delt_userm_i) / zp;
    hmult_2[i] = hmult_2_i;
    T hmult_1_i = T(0);
    if (fabs(zm) < T(1.0e-3)) {
      hmult_1_i = taylor_series_1(3, zm, delta_tau, t_delt_userm_i, user_secant);
    } else {
      hmult_1_i = user_secant * (eigentrans_i - t_delt_userm_i) / zm;
    }
    hmult_1[i] = hmult_1_i;

    T tterm = (T(1) - omega_total) * (xpos1_i + xpos2_i) / norm_saved;
    const T k1 = T(1) / eigenvalue;
    const T tcm2 = k1 * therm1;
    const T tcp2 = tcm2;
    const T tcm1 = k1 * (therm0 - tcm2);
    const T tcp1 = k1 * (therm0 + tcp2);
    const T sum_m = tcm1 + tcm2 * delta_tau;
    const T sum_p = tcp1 + tcp2 * delta_tau;
    const T tcm0 = -tcm1;
    const T tcp0 = -sum_p;
    const T t_gmult_dn = tterm * (eigentrans_i * tcm0 + sum_m);
    const T t_gmult_up = tterm * (eigentrans_i * tcp0 + tcp1);
    t_wupper0[i] = t_gmult_up * xpos2_i;
    t_wupper1[i] = t_gmult_up * xpos1_i;
    t_wlower0[i] = t_gmult_dn * xpos1_i;
    t_wlower1[i] = t_gmult_dn * xpos2_i;
    if (delta_tau <= thermal_tcutoff) {
      tterm = T(0);
      t_wupper0[i] = T(0);
      t_wupper1[i] = T(0);
      t_wlower0[i] = T(0);
      t_wlower1[i] = T(0);
    }

    const T tsgm_uu1 = tcp1 + user_stream * tcp2;
    const T tsgm_ud1 = tcm1 + user_stream * tcm2;
    const T one_minus_t_delt_userm = T(1) - t_delt_userm_i;
    const T su = tcp0 * hmult_1_i + tsgm_uu1 * one_minus_t_delt_userm -
                 tcp2 * delta_tau * t_delt_userm_i;
    const T sd = tcm0 * hmult_2_i + tsgm_ud1 * one_minus_t_delt_userm -
                 tcm2 * delta_tau * t_delt_userm_i;
    layer_tsup_up[i] = tterm * (u_xpos_i * sd + u_xneg_i * su);
  }

  const T surface_factor = T(2);
  const T surface_reflectance = use_brdf ? *brdf_f : *albedo;
  const T user_surface_reflectance = use_brdf ? *ubrdf_f : *albedo;
  solve_thermal_bvp_row(
      nlay,
      surface_reflectance,
      *emissivity,
      *surfbb,
      surface_factor,
      stream_value,
      xpos1,
      xpos2,
      eigentrans,
      t_wupper0,
      t_wupper1,
      t_wlower0,
      t_wlower1,
      lcon,
      mcon,
      col,
      elm1,
      elm2);

  if (return_fluxes) {
    const T pi = T(3.141592653589793238462643383279502884);
    const T pi2 = T(2) * pi;
    for (int level = 0; level < nlay; ++level) {
      const T down_quad = t_wupper0[level] + lcon[level] * xpos1[level] +
                          mcon[level] * xpos2[level] * eigentrans[level];
      const T up_quad = t_wupper1[level] + lcon[level] * xpos2[level] +
                        mcon[level] * xpos1[level] * eigentrans[level];
      const T up_value = do_upwelling ? pi2 * stream_value * up_quad : T(0);
      const T down_value = do_dnwelling ? pi2 * stream_value * down_quad : T(0);
      flux_up[level] = up_value;
      flux_down[level] = down_value;
      flux_net[level] = up_value - down_value;
      flux_mean[level] = (do_upwelling ? T(0.5) * up_quad : T(0)) +
                         (do_dnwelling ? T(0.5) * down_quad : T(0));
    }
    const int bottom = nlay;
    const int last = nlay - 1;
    const T down_quad =
        t_wlower0[last] + lcon[last] * xpos1[last] * eigentrans[last] + mcon[last] * xpos2[last];
    const T up_quad =
        t_wlower1[last] + lcon[last] * xpos2[last] * eigentrans[last] + mcon[last] * xpos1[last];
    const T up_value = do_upwelling ? pi2 * stream_value * up_quad : T(0);
    const T down_value = do_dnwelling ? pi2 * stream_value * down_quad : T(0);
    flux_up[bottom] = up_value;
    flux_down[bottom] = down_value;
    flux_net[bottom] = up_value - down_value;
    flux_mean[bottom] = (do_upwelling ? T(0.5) * up_quad : T(0)) +
                        (do_dnwelling ? T(0.5) * down_quad : T(0));
    for (int level = nlay - 1; level >= 0; --level) {
      const T omfac = T(1) - omega[level] * scaling[level];
      if (tau[level] * omfac <= thermal_tcutoff) {
        flux_up[level] = flux_up[level + 1];
        flux_down[level] = flux_down[level + 1];
        flux_mean[level] = flux_mean[level + 1];
        flux_net[level] = flux_up[level] - flux_down[level];
      }
    }
  }

  const int last = nlay - 1;
  const T idownsurf =
      (t_wlower0[last] + lcon[last] * xpos1[last] * eigentrans[last] +
      mcon[last] * xpos2[last]) *
      stream_value;
  T accum = surface_factor * user_surface_reflectance * idownsurf;
  if (return_profile) {
    out[nlay] = accum;
  }
  for (int n = nlay - 1; n >= 0; --n) {
    const T layer_source =
        lcon[n] * u_xpos[n] * hmult_2[n] + mcon[n] * u_xneg[n] * hmult_1[n] +
        layer_tsup_up[n];
    accum = layer_source + t_delt_userm[n] * accum;
    if (return_profile) {
      out[n] = accum;
    }
  }
  if (!return_profile) {
    out[0] = accum;
  }
}

template <typename T>
PY2SESS_HD void solar_2s_row(
    int nlay,
    T stream_value,
    T x0,
    T user_stream,
    T user_secant,
    T azmfac,
    T px11,
    T ulp,
    const T* chapman,
    const T* pxsq,
    const T* px0x,
    const T* tau,
    const T* omega,
    const T* asymm,
    const T* scaling,
    const T* albedo,
    const T* flux_factor,
    const T* brdf_f0,
    const T* brdf_f,
    const T* ubrdf_f,
    const T* slterm_isotropic,
    const T* slterm_f0,
    T* out,
    bool return_profile,
    bool return_fluxes,
    bool do_upwelling,
    bool do_dnwelling,
    bool use_brdf,
    bool use_surface_leaving,
    bool sl_isotropic_flag,
    bool plane_parallel_chapman,
    char* work) {
  const int nlev = nlay + 1;
  const int radiance_count = return_profile ? nlev : 1;
  T* flux_up = return_fluxes ? out + radiance_count : nullptr;
  T* flux_down = return_fluxes ? flux_up + nlev : nullptr;
  T* flux_net = return_fluxes ? flux_down + nlev : nullptr;
  T* flux_mean = return_fluxes ? flux_net + nlev : nullptr;
  T* delta_tau = alloc_from<T>(work, nlay);
  T* omega_total = alloc_from<T>(work, nlay);
  T* omega_asymm_3 = alloc_from<T>(work, nlay);
  T* initial_trans = alloc_from<T>(work, nlay);
  T* average_secant = alloc_from<T>(work, nlay);
  T* t_delt_mubar = alloc_from<T>(work, nlay);
  T* t_delt_userm = alloc_from<T>(work, nlay);
  T* emult_up = alloc_from<T>(work, nlay);
  T* eigentrans = alloc_from<T>(work, nlay);
  T* xpos1 = alloc_from<T>(work, nlay);
  T* xpos2 = alloc_from<T>(work, nlay);
  T* u_xpos = alloc_from<T>(work, nlay);
  T* u_xneg = alloc_from<T>(work, nlay);
  T* hmult_1 = alloc_from<T>(work, nlay);
  T* hmult_2 = alloc_from<T>(work, nlay);
  T* gamma_m = alloc_from<T>(work, nlay);
  T* aterm = alloc_from<T>(work, nlay);
  T* bterm = alloc_from<T>(work, nlay);
  T* wupper0 = alloc_from<T>(work, nlay);
  T* wupper1 = alloc_from<T>(work, nlay);
  T* wlower0 = alloc_from<T>(work, nlay);
  T* wlower1 = alloc_from<T>(work, nlay);
  T* lcon = alloc_from<T>(work, nlay);
  T* mcon = alloc_from<T>(work, nlay);
  T* col = alloc_from<T>(work, 2 * nlay);
  T* elm1 = alloc_from<T>(work, 2 * nlay > 1 ? 2 * nlay - 1 : 1);
  T* elm2 = alloc_from<T>(work, 2 * nlay > 2 ? 2 * nlay - 2 : 1);

  const T pi = T(3.141592653589793238462643383279502884);
  const T pi4 = T(4) * pi;
  const T xinv = T(1) / stream_value;
  const T hmu_stream = T(0.5) * stream_value;
  const T row_flux = *flux_factor;
  const T row_albedo = *albedo;
  const T scaled_flux = row_flux / pi4;

  bool transparent = true;
  bool has_scattering = false;
  for (int i = 0; i < nlay; ++i) {
    if (omega[i] != T(0)) {
      has_scattering = true;
    }
    const T omfac = T(1) - omega[i] * scaling[i];
    const T m1fac = T(1) - scaling[i];
    delta_tau[i] = tau[i] * omfac;
    if (delta_tau[i] != T(0)) {
      transparent = false;
    }
    omega_total[i] = clamp_value((m1fac * omega[i]) / omfac, T(1.0e-9), T(0.999999999));
    T asymm_total = clamp_value((asymm[i] - scaling[i]) / m1fac, T(-0.999999999), T(0.999999999));
    if (asymm_total >= T(0) && asymm_total < T(1.0e-9)) {
      asymm_total = T(1.0e-9);
    } else if (asymm_total < T(0) && asymm_total > T(-1.0e-9)) {
      asymm_total = T(-1.0e-9);
    }
    omega_asymm_3[i] = T(3) * omega_total[i] * asymm_total;
  }

  if (transparent) {
    const int out_count = return_profile ? nlay + 1 : 1;
    for (int i = 0; i < out_count; ++i) {
      out[i] = T(0);
    }
    if (return_fluxes) {
      const T down_value = do_dnwelling ? row_flux * x0 : T(0);
      const T mean_value = do_dnwelling ? scaled_flux : T(0);
      for (int i = 0; i < nlev; ++i) {
        flux_up[i] = T(0);
        flux_down[i] = down_value;
        flux_net[i] = -down_value;
        flux_mean[i] = mean_value;
      }
    }
    return;
  }

  T previous_tauslant = T(0);
  T cumulative_delta_tau = T(0);
  const T plane_secant = plane_parallel_chapman ? T(1) / x0 : T(0);
  int layer_pis_cutoff = nlay;
  for (int layer = 0; layer < nlay; ++layer) {
    cumulative_delta_tau += delta_tau[layer];
    T tauslant = T(0);
    if (plane_parallel_chapman) {
      tauslant = cumulative_delta_tau * plane_secant;
    } else {
      for (int k = 0; k <= layer; ++k) {
        tauslant += delta_tau[k] * chapman[k * nlay + layer];
      }
    }
    const T delta_tauslant = tauslant - previous_tauslant;
    const T user_path = delta_tau[layer] * user_secant;
    const T t_user = exp_cutoff(user_path, T(88));
    if (layer + 1 <= layer_pis_cutoff) {
      const T average = plane_parallel_chapman
                            ? plane_secant
                            : (delta_tau[layer] == T(0) ? T(0)
                                                         : delta_tauslant / delta_tau[layer]);
      const T initial = layer == 0 ? T(1) : exp(-previous_tauslant);
      const T t_mubar = exp_cutoff(delta_tauslant, T(88));
      const T sigma = average + user_secant;
      const T itrans = initial * user_secant;
      initial_trans[layer] = initial;
      average_secant[layer] = average;
      t_delt_mubar[layer] = t_mubar;
      t_delt_userm[layer] = t_user;
      emult_up[layer] = sigma == T(0) ? T(0) : itrans * (T(1) - t_mubar * t_user) / sigma;
      if (tauslant > T(88)) {
        layer_pis_cutoff = layer + 1;
      }
    } else {
      initial_trans[layer] = T(0);
      average_secant[layer] = T(0);
      t_delt_mubar[layer] = T(0);
      t_delt_userm[layer] = t_user;
      emult_up[layer] = T(0);
    }
    previous_tauslant = tauslant;
  }
  const T trans_solar_beam = previous_tauslant > T(88) ? T(0) : exp(-previous_tauslant);

  const int out_count = return_profile ? nlay + 1 : 1;
  for (int i = 0; i < out_count; ++i) {
    out[i] = T(0);
  }
  if (return_fluxes) {
    for (int i = 0; i < nlev; ++i) {
      flux_up[i] = T(0);
      flux_down[i] = T(0);
      flux_net[i] = T(0);
      flux_mean[i] = T(0);
    }
  }
  if (!has_scattering) {
    if (return_fluxes && do_dnwelling) {
      for (int level = 0; level < nlay; ++level) {
        const T direct_trans = initial_trans[level];
        const T down_value = row_flux * direct_trans * x0;
        flux_down[level] = down_value;
        flux_net[level] = -down_value;
        flux_mean[level] = scaled_flux * direct_trans;
      }
      const T down_value = row_flux * trans_solar_beam * x0;
      flux_down[nlay] = down_value;
      flux_net[nlay] = -down_value;
      flux_mean[nlay] = scaled_flux * trans_solar_beam;
    }
    return;
  }

  for (int fourier = 0; fourier <= 1; ++fourier) {
    const T surface_factor = fourier == 0 ? T(2) : T(1);
    const T delta_factor = fourier == 0 ? T(1) : T(2);
    const T add_factor = fourier == 0 ? T(1) : azmfac;
    for (int i = 0; i < nlay; ++i) {
      T sab;
      T dab;
      if (fourier == 0) {
        sab = xinv * (omega_total[i] - T(1));
        dab = xinv * (pxsq[fourier] * omega_asymm_3[i] - T(1));
      } else {
        sab = xinv * (pxsq[fourier] * omega_asymm_3[i] - T(1));
        dab = -xinv;
      }
      const T eigenvalue = sqrt(sab * dab);
      eigentrans[i] = exp_cutoff(eigenvalue * delta_tau[i], T(88));
      const T difvec = -sab / eigenvalue;
      xpos1[i] = T(0.5) * (T(1) + difvec);
      xpos2[i] = T(0.5) * (T(1) - difvec);
      const T norm_saved = stream_value * (xpos1[i] * xpos1[i] - xpos2[i] * xpos2[i]);

      if (fourier == 0) {
        const T common = T(0.5) * omega_total[i];
        const T scatter = (xpos2[i] - xpos1[i]) * hmu_stream * omega_asymm_3[i] * user_stream;
        u_xpos[i] = common + scatter;
        u_xneg[i] = common - scatter;
      } else {
        const T value = (-T(0.5) * px11 * ulp) * omega_asymm_3[i];
        u_xpos[i] = value;
        u_xneg[i] = value;
      }

      const T zp = user_secant + eigenvalue;
      const T zm = user_secant - eigenvalue;
      const T zudel = eigentrans[i] * t_delt_userm[i];
      hmult_2[i] = user_secant * (T(1) - zudel) / zp;
      if (fabs(zm) < T(1.0e-3)) {
        hmult_1[i] = taylor_series_1(3, zm, delta_tau[i], t_delt_userm[i], user_secant);
      } else {
        hmult_1[i] = user_secant * (eigentrans[i] - t_delt_userm[i]) / zm;
      }

      const bool active = i + 1 <= layer_pis_cutoff;
      const T gamma_p_raw = average_secant[i] + eigenvalue;
      const T gamma_m_raw = average_secant[i] - eigenvalue;
      gamma_m[i] = active ? gamma_m_raw : T(0);
      T cfunc = T(0);
      T dfunc = T(0);
      if (active) {
        if (fabs(gamma_m_raw) < T(1.0e-3)) {
          cfunc = taylor_series_1(3, gamma_m_raw, delta_tau[i], t_delt_mubar[i], T(1));
        } else {
          cfunc = (eigentrans[i] - t_delt_mubar[i]) / gamma_m_raw;
        }
        dfunc = (T(1) - eigentrans[i] * t_delt_mubar[i]) / gamma_p_raw;
      }
      if (fourier == 0) {
        const T common = omega_total[i] * scaled_flux;
        const T scatter = (px0x[fourier] * omega_asymm_3[i]) * (xpos1[i] - xpos2[i]) *
                          scaled_flux;
        aterm[i] = active ? (common + scatter) / norm_saved : T(0);
        bterm[i] = active ? (common - scatter) / norm_saved : T(0);
      } else {
        const T term = (px0x[fourier] * omega_asymm_3[i]) * scaled_flux / norm_saved;
        aterm[i] = active ? term : T(0);
        bterm[i] = active ? term : T(0);
      }
      const T gfunc_dn = cfunc * aterm[i] * initial_trans[i];
      const T gfunc_up = dfunc * bterm[i] * initial_trans[i];
      wupper0[i] = gfunc_up * xpos2[i];
      wupper1[i] = gfunc_up * xpos1[i];
      wlower0[i] = gfunc_dn * xpos1[i];
      wlower1[i] = gfunc_dn * xpos2[i];
    }

    T direct_beam = T(0);
    T bvp_albedo = T(0);
    T surface_reflectance = T(0);
    if (use_brdf) {
      direct_beam =
          row_flux * x0 / delta_factor / pi * trans_solar_beam * brdf_f0[fourier];
      bvp_albedo = brdf_f[fourier];
      surface_reflectance = ubrdf_f[fourier];
    } else if (fourier == 0) {
      direct_beam = row_flux * x0 / delta_factor / pi * trans_solar_beam * row_albedo;
      bvp_albedo = row_albedo;
      surface_reflectance = row_albedo;
    }
    if (use_surface_leaving) {
      const T helpv = row_flux / delta_factor;
      if (sl_isotropic_flag && fourier == 0) {
        direct_beam += (*slterm_isotropic) * helpv;
      } else {
        direct_beam += slterm_f0[fourier] * helpv;
      }
    }
    solve_bvp_row(
        nlay,
        bvp_albedo,
        direct_beam,
        surface_factor,
        stream_value,
        xpos1,
        xpos2,
        eigentrans,
        wupper0,
        wupper1,
        wlower0,
        wlower1,
        lcon,
        mcon,
        col,
        elm1,
        elm2);

    if (return_fluxes && fourier == 0) {
      const T pi2 = T(2) * pi;
      for (int level = 0; level < nlay; ++level) {
        const T down_quad = wupper0[level] + lcon[level] * xpos1[level] +
                            mcon[level] * xpos2[level] * eigentrans[level];
        const T up_quad = wupper1[level] + lcon[level] * xpos2[level] +
                          mcon[level] * xpos1[level] * eigentrans[level];
        T up_value = do_upwelling ? pi2 * stream_value * delta_factor * up_quad : T(0);
        T down_value = do_dnwelling ? pi2 * stream_value * delta_factor * down_quad : T(0);
        T mean_value = (do_upwelling ? T(0.5) * delta_factor * up_quad : T(0)) +
                       (do_dnwelling ? T(0.5) * delta_factor * down_quad : T(0));
        if (do_dnwelling) {
          down_value += row_flux * initial_trans[level] * x0;
          mean_value += scaled_flux * initial_trans[level];
        }
        flux_up[level] += up_value;
        flux_down[level] += down_value;
        flux_net[level] += up_value - down_value;
        flux_mean[level] += mean_value;
      }
      const int bottom = nlay;
      const int last = nlay - 1;
      const T down_quad =
          wlower0[last] + lcon[last] * xpos1[last] * eigentrans[last] + mcon[last] * xpos2[last];
      const T up_quad =
          wlower1[last] + lcon[last] * xpos2[last] * eigentrans[last] + mcon[last] * xpos1[last];
      T up_value = do_upwelling ? pi2 * stream_value * delta_factor * up_quad : T(0);
      T down_value = do_dnwelling ? pi2 * stream_value * delta_factor * down_quad : T(0);
      T mean_value = (do_upwelling ? T(0.5) * delta_factor * up_quad : T(0)) +
                     (do_dnwelling ? T(0.5) * delta_factor * down_quad : T(0));
      if (do_dnwelling) {
        down_value += row_flux * trans_solar_beam * x0;
        mean_value += scaled_flux * trans_solar_beam;
      }
      flux_up[bottom] += up_value;
      flux_down[bottom] += down_value;
      flux_net[bottom] += up_value - down_value;
      flux_mean[bottom] += mean_value;
    }

    const int last = nlay - 1;
    const T hom = lcon[last] * xpos1[last] * eigentrans[last] + mcon[last] * xpos2[last];
    T cumsource = surface_factor * surface_reflectance * (wlower0[last] + hom) * stream_value;
    if (return_profile) {
      out[nlay] += add_factor * delta_factor * cumsource;
    }
    for (int n = nlay - 1; n >= 0; --n) {
      T layersource = lcon[n] * u_xpos[n] * hmult_2[n] + mcon[n] * u_xneg[n] * hmult_1[n];
      if (n + 1 <= layer_pis_cutoff) {
        const T gamma_m_n = gamma_m[n];
        const T gamma_p_n = T(2) * average_secant[n] - gamma_m_n;
        T sd = (initial_trans[n] * hmult_2[n] - emult_up[n]) / gamma_m_n;
        if (fabs(gamma_m_n) < T(1.0e-3)) {
          const T itrans_userm = initial_trans[n] * user_secant;
          const T sigma_p = average_secant[n] + user_secant;
          sd = itrans_userm *
               taylor_series_2(
                   3,
                   T(1.0e-3),
                   gamma_m_n,
                   sigma_p,
                   delta_tau[n],
                   T(1),
                   t_delt_mubar[n] * t_delt_userm[n],
                   T(1));
        }
        const T su = (-initial_trans[n] * t_delt_mubar[n] * hmult_1[n] + emult_up[n]) /
                     gamma_p_n;
        layersource += u_xpos[n] * sd * aterm[n] + u_xneg[n] * su * bterm[n];
      }
      cumsource = layersource + t_delt_userm[n] * cumsource;
      if (return_profile) {
        out[n] += add_factor * delta_factor * cumsource;
      }
    }
    if (!return_profile) {
      out[0] += add_factor * delta_factor * cumsource;
    }
  }
  if (return_profile) {
    for (int level = nlay - 1; level >= 0; --level) {
      if (delta_tau[level] == T(0)) {
        out[level] = out[level + 1];
      }
    }
  }
  if (return_fluxes) {
    for (int level = nlay - 1; level >= 0; --level) {
      if (delta_tau[level] == T(0)) {
        flux_up[level] = flux_up[level + 1];
        flux_down[level] = flux_down[level + 1];
        flux_net[level] = flux_net[level + 1];
        flux_mean[level] = flux_mean[level + 1];
      }
    }
  }
}

template <typename T>
PY2SESS_HD int two_stream_flux_pair_packed_cols(int nlay) {
  return 1 + 4 * (nlay + 1);
}

template <typename T>
PY2SESS_HD void copy_flux_pair_from_packed(const T* packed, T* out, int nlay) {
  const int nlev = nlay + 1;
  const T* flux_up = packed + 1;
  const T* flux_down = flux_up + nlev;
  for (int level = 0; level < nlev; ++level) {
    out[2 * level] = flux_up[level];
    out[2 * level + 1] = flux_down[level];
  }
}

template <typename T>
PY2SESS_HD void stage_prop_layers(
    const T* prop,
    int nlay,
    int nprop,
    bool flip_layers,
    T* tau,
    T* omega,
    T* asymm,
    T* scaling) {
  for (int layer = 0; layer < nlay; ++layer) {
    const int src_layer = flip_layers ? nlay - 1 - layer : layer;
    const T* src = prop + src_layer * nprop;
    tau[layer] = src[0];
    omega[layer] = src[1];
    asymm[layer] = src[2];
    scaling[layer] = nprop > 3 ? src[3] : T(0);
  }
}

template <typename T>
PY2SESS_HD void stage_level_values(
    const T* levels,
    int nlay,
    bool flip_layers,
    T* staged) {
  const int nlev = nlay + 1;
  for (int level = 0; level < nlev; ++level) {
    staged[level] = levels[flip_layers ? nlay - level : level];
  }
}

template <typename T>
PY2SESS_HD void thermal_2s_flux_pair_row(
    int nlay,
    T stream_value,
    T user_stream,
    T thermal_tcutoff,
    const T* tau,
    const T* omega,
    const T* asymm,
    const T* scaling,
    const T* planck,
    const T* surfbb,
    const T* emissivity,
    const T* albedo,
    const T* brdf_f,
    const T* ubrdf_f,
    T* out,
    bool do_upwelling,
    bool do_dnwelling,
    bool use_brdf,
    char* work) {
  T* packed = alloc_from<T>(work, two_stream_flux_pair_packed_cols<T>(nlay));
  thermal_2s_row<T>(
      nlay,
      stream_value,
      user_stream,
      thermal_tcutoff,
      tau,
      omega,
      asymm,
      scaling,
      planck,
      surfbb,
      emissivity,
      albedo,
      brdf_f,
      ubrdf_f,
      packed,
      false,
      true,
      do_upwelling,
      do_dnwelling,
      use_brdf,
      work);
  copy_flux_pair_from_packed(packed, out, nlay);
}

template <typename T>
PY2SESS_HD void thermal_2s_prop_flux_pair_row(
    int nlay,
    int nprop,
    bool flip_layers,
    T stream_value,
    T user_stream,
    T thermal_tcutoff,
    const T* prop,
    const T* planck,
    const T* surfbb,
    const T* emissivity,
    const T* albedo,
    const T* brdf_f,
    const T* ubrdf_f,
    T* out,
    bool do_upwelling,
    bool do_dnwelling,
    bool use_brdf,
    char* work) {
  T* tau = alloc_from<T>(work, nlay);
  T* omega = alloc_from<T>(work, nlay);
  T* asymm = alloc_from<T>(work, nlay);
  T* scaling = alloc_from<T>(work, nlay);
  stage_prop_layers(prop, nlay, nprop, flip_layers, tau, omega, asymm, scaling);

  const T* planck_arg = planck;
  if (flip_layers) {
    T* staged_planck = alloc_from<T>(work, nlay + 1);
    stage_level_values(planck, nlay, true, staged_planck);
    planck_arg = staged_planck;
  }
  thermal_2s_flux_pair_row<T>(
      nlay,
      stream_value,
      user_stream,
      thermal_tcutoff,
      tau,
      omega,
      asymm,
      scaling,
      planck_arg,
      surfbb,
      emissivity,
      albedo,
      brdf_f,
      ubrdf_f,
      out,
      do_upwelling,
      do_dnwelling,
      use_brdf,
      work);
}

template <typename T>
PY2SESS_HD void solar_2s_flux_pair_row(
    int nlay,
    T stream_value,
    T x0,
    T user_stream,
    T user_secant,
    T azmfac,
    T px11,
    T ulp,
    const T* chapman,
    const T* pxsq,
    const T* px0x,
    const T* tau,
    const T* omega,
    const T* asymm,
    const T* scaling,
    const T* albedo,
    const T* flux_factor,
    const T* brdf_f0,
    const T* brdf_f,
    const T* ubrdf_f,
    const T* slterm_isotropic,
    const T* slterm_f0,
    T* out,
    bool do_upwelling,
    bool do_dnwelling,
    bool use_brdf,
    bool use_surface_leaving,
    bool sl_isotropic_flag,
    bool plane_parallel_chapman,
    char* work) {
  (void)user_stream;
  (void)user_secant;
  (void)azmfac;
  (void)px11;
  (void)ulp;

  T* delta_tau = alloc_from<T>(work, nlay);
  T* initial_trans = alloc_from<T>(work, nlay);
  T* eigentrans = alloc_from<T>(work, nlay);
  T* xpos1 = alloc_from<T>(work, nlay);
  T* xpos2 = alloc_from<T>(work, nlay);
  T* wupper0 = alloc_from<T>(work, nlay);
  T* wupper1 = alloc_from<T>(work, nlay);
  T* wlower0 = alloc_from<T>(work, nlay);
  T* wlower1 = alloc_from<T>(work, nlay);
  T* lcon = alloc_from<T>(work, nlay);
  T* mcon = alloc_from<T>(work, nlay);
  T* col = alloc_from<T>(work, 2 * nlay);
  T* elm1 = alloc_from<T>(work, 2 * nlay > 1 ? 2 * nlay - 1 : 1);
  T* elm2 = alloc_from<T>(work, 2 * nlay > 2 ? 2 * nlay - 2 : 1);

  const int nlev = nlay + 1;
  const T pi = T(3.141592653589793238462643383279502884);
  const T pi2 = T(2) * pi;
  const T pi4 = T(4) * pi;
  const T xinv = T(1) / stream_value;
  const T row_flux = *flux_factor;
  const T row_albedo = *albedo;
  const T scaled_flux = row_flux / pi4;

  bool transparent = true;
  bool has_scattering = false;
  for (int i = 0; i < nlay; ++i) {
    if (omega[i] != T(0)) {
      has_scattering = true;
    }
    const T omfac = T(1) - omega[i] * scaling[i];
    delta_tau[i] = tau[i] * omfac;
    if (delta_tau[i] != T(0)) {
      transparent = false;
    }
  }

  if (transparent) {
    const T down_value = do_dnwelling ? row_flux * x0 : T(0);
    for (int level = 0; level < nlev; ++level) {
      out[2 * level] = T(0);
      out[2 * level + 1] = down_value;
    }
    return;
  }

  T previous_tauslant = T(0);
  T cumulative_delta_tau = T(0);
  const T plane_secant = plane_parallel_chapman ? T(1) / x0 : T(0);
  int layer_pis_cutoff = nlay;
  for (int layer = 0; layer < nlay; ++layer) {
    cumulative_delta_tau += delta_tau[layer];
    T tauslant = T(0);
    if (plane_parallel_chapman) {
      tauslant = cumulative_delta_tau * plane_secant;
    } else {
      for (int k = 0; k <= layer; ++k) {
        tauslant += delta_tau[k] * chapman[k * nlay + layer];
      }
    }
    const T delta_tauslant = tauslant - previous_tauslant;
    const bool active = layer + 1 <= layer_pis_cutoff;
    const T average =
        active ? (plane_parallel_chapman
                      ? plane_secant
                      : (delta_tau[layer] == T(0) ? T(0) : delta_tauslant / delta_tau[layer]))
               : T(0);
    const T initial = active ? (layer == 0 ? T(1) : exp(-previous_tauslant)) : T(0);
    const T t_mubar = active ? exp_cutoff(delta_tauslant, T(88)) : T(0);
    initial_trans[layer] = initial;
    if (active) {
      if (tauslant > T(88)) {
        layer_pis_cutoff = layer + 1;
      }
    }
    previous_tauslant = tauslant;

    if (!has_scattering) {
      continue;
    }
    const T omfac = T(1) - omega[layer] * scaling[layer];
    const T m1fac = T(1) - scaling[layer];
    const T omega_total =
        clamp_value((m1fac * omega[layer]) / omfac, T(1.0e-9), T(0.999999999));
    T asymm_total =
        clamp_value((asymm[layer] - scaling[layer]) / m1fac, T(-0.999999999), T(0.999999999));
    if (asymm_total >= T(0) && asymm_total < T(1.0e-9)) {
      asymm_total = T(1.0e-9);
    } else if (asymm_total < T(0) && asymm_total > T(-1.0e-9)) {
      asymm_total = T(-1.0e-9);
    }
    const T omega_asymm_3 = T(3) * omega_total * asymm_total;
    const T sab = xinv * (omega_total - T(1));
    const T dab = xinv * (pxsq[0] * omega_asymm_3 - T(1));
    const T eigenvalue = sqrt(sab * dab);
    eigentrans[layer] = exp_cutoff(eigenvalue * delta_tau[layer], T(88));
    const T difvec = -sab / eigenvalue;
    xpos1[layer] = T(0.5) * (T(1) + difvec);
    xpos2[layer] = T(0.5) * (T(1) - difvec);
    const T norm_saved =
        stream_value * (xpos1[layer] * xpos1[layer] - xpos2[layer] * xpos2[layer]);

    const T gamma_p = average + eigenvalue;
    const T gamma_m = average - eigenvalue;
    T cfunc = T(0);
    T dfunc = T(0);
    if (active) {
      if (fabs(gamma_m) < T(1.0e-3)) {
        cfunc = taylor_series_1(3, gamma_m, delta_tau[layer], t_mubar, T(1));
      } else {
        cfunc = (eigentrans[layer] - t_mubar) / gamma_m;
      }
      dfunc = (T(1) - eigentrans[layer] * t_mubar) / gamma_p;
    }
    const T common = omega_total * scaled_flux;
    const T scatter = (px0x[0] * omega_asymm_3) * (xpos1[layer] - xpos2[layer]) * scaled_flux;
    const T aterm = active ? (common + scatter) / norm_saved : T(0);
    const T bterm = active ? (common - scatter) / norm_saved : T(0);
    const T gfunc_dn = cfunc * aterm * initial;
    const T gfunc_up = dfunc * bterm * initial;
    wupper0[layer] = gfunc_up * xpos2[layer];
    wupper1[layer] = gfunc_up * xpos1[layer];
    wlower0[layer] = gfunc_dn * xpos1[layer];
    wlower1[layer] = gfunc_dn * xpos2[layer];
  }
  const T trans_solar_beam = previous_tauslant > T(88) ? T(0) : exp(-previous_tauslant);

  if (!has_scattering) {
    for (int level = 0; level < nlay; ++level) {
      out[2 * level] = T(0);
      out[2 * level + 1] = do_dnwelling ? row_flux * initial_trans[level] * x0 : T(0);
    }
    out[2 * nlay] = T(0);
    out[2 * nlay + 1] = do_dnwelling ? row_flux * trans_solar_beam * x0 : T(0);
    return;
  }

  T direct_beam;
  T bvp_albedo;
  if (use_brdf) {
    direct_beam = row_flux * x0 / pi * trans_solar_beam * brdf_f0[0];
    bvp_albedo = brdf_f[0];
  } else {
    direct_beam = row_flux * x0 / pi * trans_solar_beam * row_albedo;
    bvp_albedo = row_albedo;
  }
  if (use_surface_leaving) {
    if (sl_isotropic_flag) {
      direct_beam += (*slterm_isotropic) * row_flux;
    } else {
      direct_beam += slterm_f0[0] * row_flux;
    }
  }
  solve_bvp_row(
      nlay,
      bvp_albedo,
      direct_beam,
      T(2),
      stream_value,
      xpos1,
      xpos2,
      eigentrans,
      wupper0,
      wupper1,
      wlower0,
      wlower1,
      lcon,
      mcon,
      col,
      elm1,
      elm2);

  for (int level = 0; level < nlay; ++level) {
    const T down_quad =
        wupper0[level] + lcon[level] * xpos1[level] +
        mcon[level] * xpos2[level] * eigentrans[level];
    const T up_quad =
        wupper1[level] + lcon[level] * xpos2[level] +
        mcon[level] * xpos1[level] * eigentrans[level];
    const T up_value = do_upwelling ? pi2 * stream_value * up_quad : T(0);
    T down_value = do_dnwelling ? pi2 * stream_value * down_quad : T(0);
    if (do_dnwelling) {
      down_value += row_flux * initial_trans[level] * x0;
    }
    out[2 * level] = up_value;
    out[2 * level + 1] = down_value;
  }
  const int last = nlay - 1;
  const T down_quad =
      wlower0[last] + lcon[last] * xpos1[last] * eigentrans[last] + mcon[last] * xpos2[last];
  const T up_quad =
      wlower1[last] + lcon[last] * xpos2[last] * eigentrans[last] + mcon[last] * xpos1[last];
  const T up_value = do_upwelling ? pi2 * stream_value * up_quad : T(0);
  T down_value = do_dnwelling ? pi2 * stream_value * down_quad : T(0);
  if (do_dnwelling) {
    down_value += row_flux * trans_solar_beam * x0;
  }
  out[2 * nlay] = up_value;
  out[2 * nlay + 1] = down_value;
  for (int level = nlay - 1; level >= 0; --level) {
    if (delta_tau[level] == T(0)) {
      out[2 * level] = out[2 * (level + 1)];
      out[2 * level + 1] = out[2 * (level + 1) + 1];
    }
  }
}

template <typename T>
PY2SESS_HD void solar_2s_prop_flux_pair_row(
    int nlay,
    int nprop,
    bool flip_layers,
    T stream_value,
    T x0,
    T user_stream,
    T user_secant,
    T azmfac,
    T px11,
    T ulp,
    const T* chapman,
    const T* pxsq,
    const T* px0x,
    const T* prop,
    const T* albedo,
    const T* flux_factor,
    const T* brdf_f0,
    const T* brdf_f,
    const T* ubrdf_f,
    const T* slterm_isotropic,
    const T* slterm_f0,
    T* out,
    bool do_upwelling,
    bool do_dnwelling,
    bool use_brdf,
    bool use_surface_leaving,
    bool sl_isotropic_flag,
    bool plane_parallel_chapman,
    char* work) {
  T* tau = alloc_from<T>(work, nlay);
  T* omega = alloc_from<T>(work, nlay);
  T* asymm = alloc_from<T>(work, nlay);
  T* scaling = alloc_from<T>(work, nlay);
  stage_prop_layers(prop, nlay, nprop, flip_layers, tau, omega, asymm, scaling);
  solar_2s_flux_pair_row<T>(
      nlay,
      stream_value,
      x0,
      user_stream,
      user_secant,
      azmfac,
      px11,
      ulp,
      chapman,
      pxsq,
      px0x,
      tau,
      omega,
      asymm,
      scaling,
      albedo,
      flux_factor,
      brdf_f0,
      brdf_f,
      ubrdf_f,
      slterm_isotropic,
      slterm_f0,
      out,
      do_upwelling,
      do_dnwelling,
      use_brdf,
      use_surface_leaving,
      sl_isotropic_flag,
      plane_parallel_chapman,
      work);
}

}  // namespace py2sess_native

#undef PY2SESS_HD
