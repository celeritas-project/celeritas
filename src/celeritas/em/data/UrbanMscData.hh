//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/UrbanMscData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/XsGridData.hh"

#include "CommonCoulombData.hh"

namespace celeritas
{
//! Particle categories for Urban MSC particle and material-dependent data
enum class UrbanParMatType
{
    electron = 0,
    positron,
    muhad,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Settable parameters and default values for Urban multiple scattering.
 *
 * \f$ \tau = t/\lambda \f$ where t is the true path length and \f$ \lambda \f$
 * is the mean free path of the multiple scattering. The range and safety
 * factors are used in step limitation algorithms and default values are
 * chosen to balance between simulation time and precision.
 *
<table>
  <caption>Mapping of parameter names from Geant4 to Celeritas</caption>
  <thead>
    <tr>
      <th>Geant4 Symbol</th>
      <th>Celeritas Symbol</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>dtrl</code></td><td><code>small_range_frac</code></td></tr>
    <tr><td><code>tlimitminfix</code></td><td><code>min_step</code></td></tr>
    <tr><td><code>stepmin</code></td><td><code>min_step_fallback</code></td></tr>
    <tr><td><code>tlimitminfix2</code></td><td><code>min_step_transform</code></td></tr>
    <tr><td><em>(hardcoded)</em></td><td><code>min_endpoint_energy</code></td></tr>
    <tr><td><code>tlow</code></td><td><code>min_scaling_energy</code></td></tr>
  </tbody>
</table>
 *
 * \todo Unify min_endpoint_energy with low energy limit
 * \todo Combine with lambda_limit, safety_factor in physics params
 */
struct UrbanMscParameters
{
    using Energy = units::MevEnergy;

    real_type tau_small{1e-16};  //!< small value of tau
    real_type tau_big{8};  //!< big value of tau
    real_type tau_limit{1e-6};  //!< limit of tau
    real_type safety_tol{0.01};  //!< safety tolerance
    real_type geom_limit{5e-8 * units::millimeter};  //!< minimum step
    // TODO: move these to along-step applicability
    Energy low_energy_limit{0};
    Energy high_energy_limit{0};

    //! Assume constant xs if step / range < small_range_frac ("dtrl")
    static constexpr real_type small_range_frac{0.05};

    //! For steps smaller than this, *ignore* MSC
    static constexpr real_type min_step{0.01 * units::nanometer};

    //! Minimum true path when not calculated in the step limiting
    static constexpr real_type min_step_fallback{10.0 * min_step};

    //! For steps smaller than this, true path = geometrical path
    static constexpr real_type min_step_transform{real_type{1}
                                                  * units::nanometer};

    //! Below this endpoint energy, don't sample scattering: 1 eV
    static constexpr Energy min_endpoint_energy{1e-6};

    //! The lower bound of energy to scale the minimum true path length limit
    static constexpr Energy min_scaling_energy{5e-3};
};

//---------------------------------------------------------------------------//
/*!
 * Material-dependent data for Urban MSC.
 *
 * UrbanMsc material data (see UrbanMscParams::calc_material_data) is a set of
 * precalculated material dependent parameters used in sampling the angular
 * distribution of MSC, \f$ \cos\theta \f$, and in the step limiter. The
 * coeffient vectors are used in polynomial evaluation.
 */
struct UrbanMscMaterialData
{
    using Real2 = Array<real_type, 2>;
    using Real3 = Array<real_type, 3>;

    // Step limiter
    Real2 stepmin_coeff{0, 0};  //!< Coefficients for step minimum

    // Scattering angle
    Real2 theta_coeff{0, 0};  //!< Coeffecients for theta_0 correction
    Real3 tail_coeff{0, 0, 0};  //!< Coefficients for tail parameter
    real_type tail_corr{0};  //!< Additional radiation length tail correction
};

//---------------------------------------------------------------------------//
/*!
 * Particle- and material-dependent data for MSC.
 *
 * The scaled Zeff parameters are:
 *
 *   Particle             | a    | b
 *   -------------------- | ---- | ----
 *   electron/muon/hadron | 0.87 | 2/3
 *   positron             | 0.7  | 1/2
 *
 * Two different \c d_over_r values are used: one for electrons and positrons,
 * and another for muons and hadrons.
 */
struct UrbanMscParMatData
{
    using UrbanParMatId = OpaqueId<UrbanMscParMatData>;

    real_type scaled_zeff{};  //!< a * Z^b
    real_type d_over_r{};  //!< Maximum distance/range heuristic

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const { return scaled_zeff > 0; }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for Urban MSC.
 */
template<Ownership W, MemSpace M>
struct UrbanMscData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using MaterialItems = Collection<T, W, M, PhysMatId>;
    template<class T>
    using ParticleItems = Collection<T, W, M, ParticleId>;

    //// DATA ////

    //! Particle IDs
    CoulombIds ids;
    //! Mass of of electron in MeV
    units::MevMass electron_mass;
    //! User-assignable options
    UrbanMscParameters params;
    //! Material-dependent data
    MaterialItems<UrbanMscMaterialData> material_data;
    //! Number of particles this model applies to
    ParticleId::size_type num_particles;
    //! Number of particle categories for particle and material-dependent data
    ParticleId::size_type num_par_mat;
    //! Map from particle ID to index in particle and material-dependent data
    ParticleItems<UrbanMscParMatData::UrbanParMatId> pid_to_pmdata;
    //! Map from particle ID to index in cross sections
    ParticleItems<MscParticleId> pid_to_xs;
    //! Particle and material-dependent data
    Items<UrbanMscParMatData> par_mat_data;  //!< [mat][particle]
    //! Scaled xs data
    Items<UniformGridRecord> xs;  //!< [mat][particle]

    // Backend storage
    Items<real_type> reals;

    //// METHODS ////

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && electron_mass > zero_quantity() && !material_data.empty()
               && num_particles >= 2 && num_par_mat >= 2
               && !pid_to_pmdata.empty() && !pid_to_xs.empty()
               && !par_mat_data.empty() && !xs.empty() && !reals.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    UrbanMscData& operator=(UrbanMscData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        ids = other.ids;
        electron_mass = other.electron_mass;
        params = other.params;
        material_data = other.material_data;
        num_particles = other.num_particles;
        num_par_mat = other.num_par_mat;
        pid_to_pmdata = other.pid_to_pmdata;
        pid_to_xs = other.pid_to_xs;
        par_mat_data = other.par_mat_data;
        xs = other.xs;
        reals = other.reals;
        return *this;
    }
};

}  // namespace celeritas
