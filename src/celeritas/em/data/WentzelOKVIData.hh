//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/WentzelOKVIData.hh
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
 * Parameters used in both single and multiple Coulomb scattering models.
 *
 * The Wentzel VI MSC model is used to sample scatterings with angles below the
 * polar angle limit, and the single Coulomb scattering model is used for
 * angles above the limit.
 */
struct CoulombParameters
{
    //! Whether to use combined single and multiple scattering
    bool is_combined;
    //! Polar angle limit between single and multiple scattering
    real_type costheta_limit;
    //! Factor for the screening coefficient
    real_type screening_factor;
    //! Factor used to calculate the maximum scattering angle off of a nucleus
    real_type a_sq_factor;
    // Model for the form factor to use
    NuclearFormFactorType form_factor_type;

    explicit CELER_FUNCTION operator bool() const
    {
        return costheta_limit >= -1 && costheta_limit <= 1
               && screening_factor > 0 && a_sq_factor > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Per-element data used by the Coulomb scattering and Wentzel VI models.
 *
 * The matrix of coefficients used to approximate the ratio of the Mott to
 * Rutherford cross sections was developed in
 * T. Lijian, H. Quing and L. Zhengming, Radiat. Phys. Chem. 45 (1995),
 *   235-245
 * and
 * M. J. Boschini et al. arXiv:1111.4042
 */
struct MottElementData
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
 * Constant shared data used by the Coulomb scattering and Wentzel VI models.
 */
template<Ownership W, MemSpace M>
struct WentzelOKVIData
{
    template<class T>
    using ElementItems = celeritas::Collection<T, W, M, ElementId>;
    template<class T>
    using IsotopeItems = celeritas::Collection<T, W, M, IsotopeId>;

    // User-assignable parameters
    CoulombParameters params;

    // Constant prefactor for the squared momentum transfer [(MeV/c)^-2]
    IsotopeItems<real_type> nuclear_form_prefactor;

    // Per element form factors
    ElementItems<MottElementData> elem_data;

    // Check if the data is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        return params && !elem_data.empty();
    }

    template<Ownership W2, MemSpace M2>
    WentzelOKVIData& operator=(WentzelOKVIData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        params = other.params;
        nuclear_form_prefactor = other.nuclear_form_prefactor;
        elem_data = other.elem_data;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
