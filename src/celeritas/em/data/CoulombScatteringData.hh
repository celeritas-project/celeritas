//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/CoulombScatteringData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Particle and action ids used by CoulombScatteringModel.
 */
struct CoulombScatteringIds
{
    ActionId action;
    ParticleId electron;
    ParticleId positron;
    ParticleId proton;

    explicit CELER_FUNCTION operator bool() const
    {
        // TODO: enable when protons are supported
        return action && electron && positron /* && proton */;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Per-element data used by the CoulombScatteringModel.
 *
 * The matrix of coefficients used to approximate the ratio of the Mott to
 * Rutherford cross sections was developed in
 * T. Lijian, H. Quing and L. Zhengming, Radiat. Phys. Chem. 45 (1995),
 *   235-245
 * and
 * M. J. Boschini et al. arXiv:1111.4042
 */
struct CoulombScatteringElementData
{
    //!@{
    //! \name Dimensions for Mott coefficient matrices
    static constexpr size_type num_mott_beta_bins = 6;
    static constexpr size_type num_mott_theta_bins = 5;
    static constexpr size_type num_mott_elements = 92;
    //!@}

    using BetaArray = Array<real_type, num_mott_beta_bins>;
    using ThetaArray = Array<real_type, num_mott_theta_bins>;
    using MottCoeffMatrix = Array<BetaArray, num_mott_theta_bins>;

    //! Matrix of Mott coefficients [theta][beta]
    MottCoeffMatrix mott_coeff;
};

//---------------------------------------------------------------------------//
/*!
 * Supported models of nuclear form factors.
 */
enum class NuclearFormFactorType
{
    none,
    flat,
    exponential,
    gaussian
};

//---------------------------------------------------------------------------//
/*!
 * Settable parameters and default values for single Coulomb scattering.
 *
 * For single Coulomb scattering, \c costheta_limit will be 1 (\f$ \cos
 * \theta_{\text{min}} \f$). If single scattering is used in combination with
 * multiple scattering, it will be \f$ \cos \theta_{\text{max}} \f$.
 */
struct CoulombScatteringParameters
{
    bool is_combined;  //!< Use combined single and multiple scattering
    real_type costheta_limit;  //!< Limit on the cos of the scattering angle
    real_type a_sq_factor;  //!< Factor used to calculate the maximum
                            //!< scattering angle off of a nucleus
    real_type screening_factor;  //!< Factor for the screening coefficient
    NuclearFormFactorType form_factor_type;  //!< Model for the form factor
};

//---------------------------------------------------------------------------//
/*!
 * Constant shared data used by the CoulombScatteringModel.
 */
template<Ownership W, MemSpace M>
struct CoulombScatteringData
{
    template<class T>
    using MaterialItems = celeritas::Collection<T, W, M, MaterialId>;
    template<class T>
    using ElementItems = celeritas::Collection<T, W, M, ElementId>;
    template<class T>
    using IsotopeItems = celeritas::Collection<T, W, M, IsotopeId>;

    // Ids
    CoulombScatteringIds ids;

    // User-assignable options
    CoulombScatteringParameters params;

    //! Constant prefactor for the squared momentum transfer [(MeV/c)^-2]
    IsotopeItems<real_type> nuclear_form_prefactor;

    // Per element form factors
    ElementItems<CoulombScatteringElementData> elem_data;

    // Inverse of the effective atomic mass to the 2/3 power
    MaterialItems<real_type> ma_cbrt_sq_inv;

    // Check if the data is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && !nuclear_form_prefactor.empty() && !elem_data.empty()
               && params.is_combined == !ma_cbrt_sq_inv.empty();
    }

    // Copy initialize from an existing CoulombScatteringData
    template<Ownership W2, MemSpace M2>
    CoulombScatteringData& operator=(CoulombScatteringData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        ids = other.ids;
        params = other.params;
        nuclear_form_prefactor = other.nuclear_form_prefactor;
        elem_data = other.elem_data;
        ma_cbrt_sq_inv = other.ma_cbrt_sq_inv;
        return *this;
    }
};

using CoulombScatteringDeviceRef = DeviceCRef<CoulombScatteringData>;
using CoulombScatteringHostRef = HostCRef<CoulombScatteringData>;
using CoulombScatteringRef = NativeCRef<CoulombScatteringData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
