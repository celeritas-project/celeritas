//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/WentzelVIMscData.hh
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
//---------------------------------------------------------------------------//
/*!
 * Settable parameters and default values for Wentzel VI multiple scattering.
 */
struct WentzelVIMscParameters
{
    using Energy = units::MevEnergy;

    real_type single_scattering_factor{1.25};
    Energy low_energy_limit{0};
    Energy high_energy_limit{0};

    //! The minimum value of the true path length limit: 1 nm
    static CELER_CONSTEXPR_FUNCTION real_type limit_min_fix()
    {
        return 1e-7 * units::centimeter;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for Wentzel VI MSC.
 */
template<Ownership W, MemSpace M>
struct WentzelVIMscData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ParticleItems = Collection<T, W, M, ParticleId>;

    //// DATA ////

    //! Particle IDs
    CoulombIds ids;
    //! Mass of electron in MeV
    units::MevMass electron_mass;
    //! User-assignable options
    WentzelVIMscParameters params;
    //! Number of particles this model applies to
    ParticleId::size_type num_particles;
    //! Map from particle ID to index in cross sections
    ParticleItems<MscParticleId> pid_to_xs;
    //! Scaled xs data
    Items<XsGridData> xs;  //!< [mat][particle]

    // Backend storage
    Items<real_type> reals;

    //// METHODS ////

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && electron_mass > zero_quantity() && num_particles >= 2
               && !pid_to_xs.empty() && !xs.empty() && !reals.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    WentzelVIMscData& operator=(WentzelVIMscData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        ids = other.ids;
        electron_mass = other.electron_mass;
        params = other.params;
        num_particles = other.num_particles;
        pid_to_xs = other.pid_to_xs;
        xs = other.xs;
        reals = other.reals;
        return *this;
    }
};

}  // namespace celeritas
