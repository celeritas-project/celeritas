//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SeltzerBergerInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/StackAllocator.hh"
#include "base/Types.hh"
#include "physics/base/CutoffView.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Types.hh"
#include "physics/base/Units.hh"
#include "physics/grid/TwodGridInterface.hh"
#include "physics/material/ElementView.hh"
#include "physics/material/MaterialView.hh"
#include "physics/material/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Seltzer-Berger differential cross section tables for a single element.
 *
 * The 2D grid data is organized by log E on the x axis and fractional exiting
 * energy (0 to 1) on the y axis. The values are in millibarns, but their
 * magnitude isn't important since we always take ratios.
 *
 * \c argmax is the y index of the largest cross section at a given incident
 * energy point.
 *
 * \todo We could use way smaller integers for argmax, even i/j here, because
 * these tables are so small.
 */
struct SBElementTableData
{
    using EnergyUnits = units::LogMev;
    using XsUnits     = units::Millibarn;

    TwodGridData         grid;   //!< Cross section grid and data
    ItemRange<size_type> argmax; //!< Y index of the largest XS for each energy

    explicit inline CELER_FUNCTION operator bool() const
    {
        return grid && argmax.size() == grid.x.size();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Bremsstrahlung differential cross section (DCS) data for SB sampling.
 *
 * The value grids are organized per element ID, and each 2D grid is:
 * - x: logarithm of the energy [MeV] of the incident charged dparticle
 * - y: ratio of exiting photon energy to incident particle energy
 * - value: differential cross section (microbarns)
 */
template<Ownership W, MemSpace M>
struct SeltzerBergerTableData
{
    //// MEMBER FUNCTIONS ////

    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ElementItems = Collection<T, W, M, ElementId>;

    //// MEMBER DATA ////

    Items<real_type>                 reals;
    Items<size_type>                 sizes;
    ElementItems<SBElementTableData> elements;

    //// MEMBER FUNCTIONS ////

    //! Whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !reals.empty() && !sizes.empty() && !elements.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SeltzerBergerTableData&
    operator=(const SeltzerBergerTableData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        reals    = other.reals;
        sizes    = other.sizes;
        elements = other.elements;
        return *this;
    }
};

//! Helper struct for making assignment easier
struct SeltzerBergerIds
{
    //! Model ID
    ModelId model;
    //! ID of an electron
    ParticleId electron;
    //! ID of an positron
    ParticleId positron;
    //! ID of a gamma
    ParticleId gamma;

    //! Whether the IDs are assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return model && electron && positron && gamma;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for sampling SeltzerBergerInteractor.
 */
template<Ownership W, MemSpace M>
struct SeltzerBergerData
{
    using MevMass = units::MevMass;
    //// MEMBER DATA ////

    //! IDs in a separate struct for readability/easier copying
    SeltzerBergerIds ids;

    //! Electron mass [MeV / c^2]
    MevMass electron_mass;

    // Differential cross section storage
    SeltzerBergerTableData<W, M> differential_xs;

    //// MEMBER FUNCTIONS ////

    //! Whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return ids && electron_mass.value() > 0 && differential_xs;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SeltzerBergerData& operator=(const SeltzerBergerData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        ids             = other.ids;
        electron_mass   = other.electron_mass;
        differential_xs = other.differential_xs;
        return *this;
    }
};

using SeltzerBergerDeviceRef
    = SeltzerBergerData<Ownership::const_reference, MemSpace::device>;
using SeltzerBergerHostRef
    = SeltzerBergerData<Ownership::const_reference, MemSpace::host>;
using SeltzerBergerNativeRef
    = SeltzerBergerData<Ownership::const_reference, MemSpace::native>;

//---------------------------------------------------------------------------//
/*!
 * Seltzer-Berger model for electron and positron bremsstrahlung processes.
 *
 * Given an incoming electron or positron of sufficient energy (as per
 * CutOffView), this class provides the energy loss of these particles due to
 * radiation of photons in the field of a nucleus. This model improves accuracy
 * using cross sections based on interpolation of published tables from Seltzer
 * and Berger given in Nucl. Instr. and Meth. in Phys. Research B, 12(1):95–134
 * (1985) and Atomic Data and Nuclear Data Tables, 35():345 (1986). The cross
 * sections are obtained from SBEnergyDistribution and are appropriately scaled
 * in the case of positrons via SBPositronXsCorrector (to be done).
 *
 * \note This interactor performs an analogous sampling as in Geant4's
 * G4SeltzerBergerModel, documented in 10.2.1 of the Geant Physics Reference
 * (release 10.6). The implementation is based on Geant4 10.4.3.
 */
class SeltzerBergerInteractor
{
  public:
    //!@{
    //! Type aliases
    using Energy   = units::MevEnergy;
    using Momentum = units::MevMomentum;
    //!@}

  public:
    //! Construct sampler from device/shared and state data
    inline CELER_FUNCTION
    SeltzerBergerInteractor(const SeltzerBergerNativeRef& shared,
                            const ParticleTrackView&      particle,
                            const Real3&                  inc_direction,
                            const CutoffView&             cutoffs,
                            StackAllocator<Secondary>&    allocate,
                            const MaterialView&           material,
                            const ElementComponentId&     elcomp_id);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    // Device (host CPU or GPU device) references
    const SeltzerBergerNativeRef& shared_;
    // Incident particle energy
    const Energy inc_energy_;
    // Incident particle direction
    const Momentum inc_momentum_;
    // Incident particle direction
    const Real3& inc_direction_;
    // Incident particle flag for selecting XS correction factor
    const bool inc_particle_is_electron_;
    // Production cutoff for gammas
    const Energy gamma_cutoff_;
    // Allocate space for a secondary particle
    StackAllocator<Secondary>& allocate_;
    // Material in which interaction occurs
    const MaterialView& material_;
    // Element in which interaction occurs
    const ElementComponentId elcomp_id_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "SeltzerBergerInteractor.i.hh"
