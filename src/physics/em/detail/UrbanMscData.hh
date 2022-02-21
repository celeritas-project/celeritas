//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UrbanMscData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "base/Macros.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * UrbanMscModel settable parameters
 */
struct UrbanMscParameters
{
    real_type tau_small{1e-16};   //!< small value of tau
    real_type tau_big{8};         //!< big value of tau
    real_type tau_limit{1e-6};    //!< limit of tau
    real_type lambda_limit{0.01}; //!< lambda limit (default: 1 mm)
    real_type range_fact{0.04};   //!< range_factor for e-/e+ (0.02 for mu/h)
    real_type safety_fact{0.6};   //!< safety factor
};

//---------------------------------------------------------------------------//
/*!
 * UrbanMsc material data: G4UrbanMscModel::mscData
 */
struct UrbanMscMaterialData
{
    real_type zeff;     //!< effective atomic_number
    real_type z23;      //!< ipow<4>(log(zeff)/6)
    real_type coeffth1; //!< correction in theta_0 formula
    real_type coeffth2; //!< correction in theta_0 formula
    real_type coeffc1;  //!< coefficient of tail parameters
    real_type coeffc2;  //!< coefficient of tail parameters
    real_type coeffc3;  //!< coefficient of tail parameters
    real_type coeffc4;  //!< coefficient of tail parameters
    real_type stepmina; //!< coefficient of the step minimum calculation
    real_type stepminb; //!< coefficient of the step minimum calculation
    real_type doverra;  //!< distance/range for e-/e+
    real_type doverrb;  //!< distance/range for mu/h
};

//---------------------------------------------------------------------------//
/*!
 * Device data for step limitation algorithms and angular scattering.
 */
template<Ownership W, MemSpace M>
struct UrbanMscData
{
    //! Model ID
    ModelId model_id;

    //! ID of a electron
    ParticleId electron_id;

    //! ID of a positron
    ParticleId positron_id;

    //! Mass of of electron in MeV
    units::MevMass electron_mass;

    UrbanMscParameters params;

    template<class T>
    using ElementItems = celeritas::Collection<T, W, M, MaterialId>;
    ElementItems<UrbanMscMaterialData> msc_data;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return model_id && electron_id && positron_id && !msc_data.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    UrbanMscData& operator=(const UrbanMscData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        model_id      = other.model_id;
        electron_id   = other.electron_id;
        positron_id   = other.positron_id;
        electron_mass = other.electron_mass;
        params        = other.params;
        msc_data      = other.msc_data;
        return *this;
    }
};

using UrbanMscDeviceRef
    = UrbanMscData<Ownership::const_reference, MemSpace::device>;
using UrbanMscHostRef
    = UrbanMscData<Ownership::const_reference, MemSpace::host>;
using UrbanMscNativeRef
    = UrbanMscData<Ownership::const_reference, MemSpace::native>;

} // namespace detail
} // namespace celeritas
