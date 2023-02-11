//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Settable parameters and default values for Urban multiple scattering.
 *
 * \f$ \tau = t/\lambda \f$ where t is the true path length and \f$ \lambda \f$
 * is the mean free path of the multiple scattering. The range and safety
 * factors are used in step limitation algorithms and default values are
 * chosen to balance between simulation time and precision.
 */
struct UrbanMscParameters
{
    using Energy = units::MevEnergy;

    real_type tau_small{1e-16};  //!< small value of tau
    real_type tau_big{8};  //!< big value of tau
    real_type tau_limit{1e-6};  //!< limit of tau
    real_type lambda_limit{1 * units::millimeter};  //!< lambda limit
    real_type range_fact{0.04};  //!< range_factor for e-/e+ (0.2 for muon/h)
    real_type safety_fact{0.6};  //!< safety factor
    real_type safety_tol{0.01};  //!< safety tolerance
    real_type geom_limit{5e-8 * units::millimeter};  //!< minimum step
    Energy low_energy_limit{1e-5};  //!< 10 eV
    Energy high_energy_limit{1e+2};  //!< 100 MeV

    //! A scale factor for the range
    static CELER_CONSTEXPR_FUNCTION real_type dtrl() { return 5e-2; }

    //! The minimum value of the true path length limit: 0.01 nm
    static CELER_CONSTEXPR_FUNCTION real_type limit_min_fix()
    {
        return 1e-9 * units::centimeter;
    }

    //! Default minimum of the true path limit: 1e-8 cm
    static CELER_CONSTEXPR_FUNCTION real_type limit_min()
    {
        return 10 * limit_min_fix();
    }

    //! For steps below this value, true = geometrical (no MSC to be applied)
    static CELER_CONSTEXPR_FUNCTION real_type min_step()
    {
        return 100 * limit_min_fix();
    }

    //! Below this endpoint energy, don't sample scattering: 1 eV
    static CELER_CONSTEXPR_FUNCTION Energy min_sampling_energy()
    {
        return units::MevEnergy{1e-6};
    }

    //! The lower bound of energy to scale the minimum true path length limit
    static CELER_CONSTEXPR_FUNCTION Energy min_scaling_energy()
    {
        return units::MevEnergy(5e-3);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Material-dependent data for Urban MSC.
 *
 * UrbanMsc material data (see G4UrbanMscModel::mscData) is a set of
 * precalculated material dependent parameters used in sampling the angular
 * distribution of MSC, \f$ \cos\theta \f$. All parameters are unitless.
 */
struct UrbanMscMaterialData
{
    using Real4 = Array<real_type, 4>;

    real_type coeffth1{};  //!< correction in theta_0 formula
    real_type coeffth2{};  //!< correction in theta_0 formula
    Real4 d{0, 0, 0, 0};  //!< coefficients of tail parameters
    real_type stepmin_a{};  //!< coefficient of the step minimum calculation
    real_type stepmin_b{};  //!< coefficient of the step minimum calculation
    real_type d_over_r{};  //!< the maximum distance/range for e-/e+
    real_type d_over_r_mh{};  //!< the maximum distance/range for muon/h
};

//---------------------------------------------------------------------------//
/*!
 * Physics IDs for MSC.
 *
 * TODO these will probably be changed to a map over all particle IDs.
 */
struct UrbanMscIds
{
    ParticleId electron;
    ParticleId positron;

    //! Whether the IDs are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return electron && positron;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Particle- and material-dependent data for MSC.
 *
 * The scaled Zeff parameters are:
 *
 *   Particle | a    | b
 *   -------- | ---- | ----
 *   electron | 0.87 | 2/3
 *   positron | 0.7  | 1/2
 */
struct UrbanMscParMatData
{
    XsGridData xs;  //!< For calculating MFP
    real_type scaled_zeff{};  //!< a * Z^b

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return xs && scaled_zeff > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for Urban MSC.
 *
 * Since the model currently applies only to electrons and positrons, the
 * particles are hardcoded to be length 2. TODO: extend to other charged
 * particles when further physics is implemented.
 */
template<Ownership W, MemSpace M>
struct UrbanMscData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using MaterialItems = celeritas::Collection<T, W, M, MaterialId>;

    //// DATA ////

    //! Particle IDs
    UrbanMscIds ids;
    //! Mass of of electron in MeV
    units::MevMass electron_mass;
    //! User-assignable options
    UrbanMscParameters params;
    //! Material-dependent data
    MaterialItems<UrbanMscMaterialData> material_data;
    //! Particle and material-dependent data
    Items<UrbanMscParMatData> par_mat_data;  // [mat]{electron, positron}

    // Backend storage
    Items<real_type> reals;

    //// METHODS ////

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && electron_mass > zero_quantity() && !material_data.empty()
               && !par_mat_data.empty() && !reals.empty();
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
        par_mat_data = other.par_mat_data;
        reals = other.reals;
        return *this;
    }

    //! Get the data location for a material + particle
    CELER_FUNCTION ItemId<UrbanMscParMatData>
    at(MaterialId mat, ParticleId par) const
    {
        CELER_EXPECT(mat && par);
        size_type result = mat.unchecked_get() * 2;
        result += (par == this->ids.electron ? 0 : 1);
        CELER_ENSURE(result < this->par_mat_data.size());
        return ItemId<UrbanMscParMatData>{result};
    }
};

}  // namespace celeritas
