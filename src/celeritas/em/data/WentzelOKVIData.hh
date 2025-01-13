//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/WentzelOKVIData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Parameters used in both single Coulomb scattering and Wentzel VI MSC models.
 *
 * When the single Coulomb scattering and Wentzel VI MSC models are used
 * together, the MSC model is used to sample scatterings with angles below the
 * polar angle limit, and the single scattering model is used for angles above
 * the limit.
 */
struct CoulombParameters
{
    //! Whether to use combined single and multiple scattering
    bool is_combined{};
    //! Polar angle limit between single and multiple scattering
    real_type costheta_limit{};
    //! Factor for the screening coefficient
    real_type screening_factor{};
    //! Factor used to calculate the maximum scattering angle off of a nucleus
    real_type a_sq_factor{};
    // Model for the form factor to use
    NuclearFormFactorType form_factor_type{NuclearFormFactorType::exponential};

    explicit CELER_FUNCTION operator bool() const
    {
        return costheta_limit >= -1 && costheta_limit <= 1
               && screening_factor > 0 && a_sq_factor >= 0
               && form_factor_type != NuclearFormFactorType::size_;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Per-element data used by the Coulomb scattering and Wentzel VI models.
 *
 * The matrix of coefficients used to approximate the ratio of the Mott to
 * Rutherford cross sections was developed in \citet{lijian-mott-1995,
 * https://doi.org/10.1016/0969-806X(94)00063-8}. Using the same procedure as
 * in Lijian, the coefficients were extended in \citet{boschini-mott-1993,
 * https://doi.org/10.1016/j.radphyschem.2013.04.020} to include positrons and
 * the interaction of electrons and positrons with higher Z nuclei (1 <= Z <=
 * 118).
 */
struct MottElementData
{
    //!@{
    //! \name Dimensions for Mott coefficient matrices
    static constexpr size_type num_beta = 6;
    static constexpr size_type num_theta = 5;
    static constexpr size_type num_elements = 118;
    //!@}

    using BetaArray = Array<real_type, num_beta>;
    using ThetaArray = Array<real_type, num_theta>;
    using MottCoeffMatrix = Array<BetaArray, num_theta>;

    //! Matrix of Mott coefficients [theta][beta]
    MottCoeffMatrix electron;
    MottCoeffMatrix positron;
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
    template<class T>
    using MaterialItems = Collection<T, W, M, MaterialId>;

    // User-assignable parameters
    CoulombParameters params;

    // Constant prefactor for the squared momentum transfer [(MeV/c)^{-2}]
    IsotopeItems<real_type> nuclear_form_prefactor;

    // Per element form factors
    ElementItems<MottElementData> mott_coeffs;

    // Inverse effective A^2/3 [1/mass^2/3]
    MaterialItems<real_type> inv_mass_cbrt_sq;

    // Check if the data is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        return params && !mott_coeffs.empty()
               && params.is_combined == !inv_mass_cbrt_sq.empty();
    }

    template<Ownership W2, MemSpace M2>
    WentzelOKVIData& operator=(WentzelOKVIData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        params = other.params;
        nuclear_form_prefactor = other.nuclear_form_prefactor;
        mott_coeffs = other.mott_coeffs;
        inv_mass_cbrt_sq = other.inv_mass_cbrt_sq;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
