//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/FluctuationData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Material-dependent parameters used in the energy loss fluctuation model.
 */
struct UrbanFluctuationParameters
{
    using Energy = units::MevEnergy;
    using Real2 = Array<real_type, 2>;

    Real2 binding_energy;  //!< Binding energies E_1 and E_2 [MeV]
    Real2 log_binding_energy;  //!< Log of binding energies [LogMevEnergy]
    Real2 oscillator_strength;  //!< Oscillator strengths f_1 and f_2
};

//---------------------------------------------------------------------------//
/*!
 * Data needed to sample from the energy loss distribution.
 */
template<Ownership W, MemSpace M>
struct FluctuationData
{
    template<class T>
    using MaterialItems = Collection<T, W, M, PhysMatId>;
    using Mass = units::MevMass;

    //// MEMBER DATA ////

    ParticleId electron_id;  //!< ID of an electron
    Mass electron_mass;  //!< Electron mass
    MaterialItems<UrbanFluctuationParameters> urban;  //!< Model parameters

    //// MEMBER FUNCTIONS ////

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return electron_id && electron_mass > zero_quantity();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    FluctuationData& operator=(FluctuationData<W2, M2> const& other)
    {
        electron_id = other.electron_id;
        electron_mass = other.electron_mass;
        urban = other.urban;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
