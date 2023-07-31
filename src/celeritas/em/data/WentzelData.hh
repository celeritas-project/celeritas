//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/WentzelData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/model/detail/MottInterpolatedCoefficients.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Particle and action ids used by WentzelModel
 */
struct WentzelIds
{
    ActionId action;
    ParticleId electron;
    ParticleId positron;

    explicit CELER_FUNCTION operator bool() const
    {
        return action && electron && positron;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Per-element data used by the WentzelModel
 *
 * The matrix of coefficients used to approximate the ratio of the Mott to
 * Rutherford cross sections was developed in
 * T. Lijian, H. Quing and L. Zhengming, Radiat. Phys. Chem. 45 (1995),
 *   235-245
 * and
 * M. J. Boschini et al. arXiv:1111.4042
 */
struct WentzelElementData
{
    using BetaArray = Array<real_type, detail::num_mott_beta_bins>;
    using ThetaArray = Array<real_type, detail::num_mott_theta_bins>;

    //! Matrix of Mott coefficients [theta][beta]
    Array<BetaArray, detail::num_mott_theta_bins> mott_coeff;
};

//---------------------------------------------------------------------------//
/*!
 * Supported models of nuclear form factors
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
 * Constant shared data used by the WentzelModel
 */
template<Ownership W, MemSpace M>
struct WentzelData
{
    template<class T>
    using ElementItems = celeritas::Collection<T, W, M, ElementId>;

    // Ids
    WentzelIds ids;

    // Per element form factors
    ElementItems<WentzelElementData> elem_data;

    // User-defined factor for the screening coefficient
    real_type screening_factor;

    // Model for the form factor to use
    NuclearFormFactorType form_factor_type;

    // Check if the data is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && !elem_data.empty();
    }

    // Copy initialize from an existing WentzelData
    template<Ownership W2, MemSpace M2>
    WentzelData& operator=(WentzelData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        ids = other.ids;
        elem_data = other.elem_data;
        form_factor_type = other.form_factor_type;
        screening_factor = other.screening_factor;
        return *this;
    }
};

using WentzelDeviceRef = DeviceCRef<WentzelData>;
using WentzelHostRef = HostCRef<WentzelData>;
using WentzelRef = NativeCRef<WentzelData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
