//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/woodcock/detail/MajorantBuilder.hh
//! \sa celeritas/woodcock/Woodcock.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/StateDataStore.hh"
#include "corecel/inp/Grid.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/PhysicsData.hh"

namespace celeritas
{
class MaterialParams;
class ParticleParams;
class PhysicsParams;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Tabulate a cross section that bounds the total over a set of materials.
 *
 * The majorant at energy \em E over materials \em m is
 * \f[
   \Sigma^*(E) \ge \max_m \sum_p \Sigma_{m,p}(E),
   \f]
 * where the macroscopic cross sections of the processes \em p are evaluated
 * exactly as during transport. (They might be interpolated from tables or
 * calculated on the fly, and they can be discontinuous.)
 *
 * The majorant is tabulated on a uniform log energy grid and is linearly
 * interpolated on energy. The value at each grid point is the
 * maximum over materials at that energy. Since linear interpolation is an
 * underestimate for concave or discontinuous functions, each bin is then
 * sampled on a finer grid: if a cross section in the bin exceeds the
 * interpolated majorant, the values on both sides of the bin are set to the
 * maximum sampled in the bin. A small global safety factor accounts for
 * variations between the sampled energies.
 *
 * If the cross section were to exceed the majorant, the null collision
 * coefficient would be negative and the collision probability would be
 * greater than one. Sampling without bias is still possible with a different
 * collision probability and statistical weights (Eqs. 11 and 12 of
 * \cite{galtier-integralformulation-2013} ), but this increases the variance
 * and is not implemented.
 */
class MajorantBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using VecMat = std::vector<PhysMatId>;
    //!@}

    //! Number of grid bins per decade of energy
    static constexpr size_type bins_per_decade() { return 128; }
    //! Number of cross section samples per grid bin
    static constexpr size_type samples_per_bin() { return 16; }
    //! Multiplicative safety factor
    static constexpr double safety_factor() { return 1.005; }

  public:
    // Construct with shared data and lower energy
    MajorantBuilder(MaterialParams const& materials,
                    ParticleParams const& particles,
                    PhysicsParams const& physics,
                    ParticleId particle,
                    Energy min_energy);

    // Build the majorant for the given materials
    inp::UniformGrid operator()(VecMat const& materials);

    // Calculate the total macroscopic cross section [1/len]
    real_type calc_xs(PhysMatId material, Energy energy);

    //! Highest tabulated energy
    Energy max_energy() const { return max_energy_; }

  private:
    template<template<Ownership, MemSpace> class S>
    using HostStore = StateDataStore<S, MemSpace::host>;

    MaterialParams const& materials_;
    ParticleParams const& particles_;
    PhysicsParams const& physics_;
    ParticleId particle_;
    Energy min_energy_;
    Energy max_energy_;

    // Single-track states for evaluating cross sections
    HostStore<ParticleStateData> particle_state_;
    HostStore<PhysicsStateData> physics_state_;

    double calc_max_xs(VecMat const& materials, double energy);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
