//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/data/DTMixMucfData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <utility>

#include "corecel/Macros.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "corecel/io/Join.hh"
#include "celeritas/Types.hh"
#include "celeritas/mucf/Types.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * ParticleIds used by the \c DTMixMucfModel .
 */
struct MucfParticleIds
{
    //! Primary
    ParticleId mu_minus;

    //!@{
    //! Elementary particles and nuclei
    ParticleId proton;
    ParticleId triton;
    ParticleId neutron;
    ParticleId alpha;
    ParticleId he3;
    //!@}

    //!@{
    //! Muonic atoms
    ParticleId muonic_hydrogen;
    ParticleId muonic_deuteron;
    ParticleId muonic_triton;
    ParticleId muonic_alpha;
    ParticleId muonic_he3;
    //!@}

    //! Check whether all particles are assigned
    CELER_FUNCTION explicit operator bool() const
    {
        return mu_minus && proton && triton && neutron && alpha && he3
               && muonic_hydrogen && muonic_deuteron && muonic_triton
               && muonic_alpha && muonic_he3;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Particle masses used by the \c DTMixMucfModel .
 */
struct MucfParticleMasses
{
    //! Primary
    units::MevMass mu_minus;

    //!@{
    //! Elementary particles and nuclei
    units::MevMass proton;
    units::MevMass triton;
    units::MevMass neutron;
    units::MevMass alpha;
    units::MevMass he3;
    //!@}

    //!@{
    //! Muonic atoms
    units::MevMass muonic_hydrogen;
    units::MevMass muonic_deuteron;
    units::MevMass muonic_triton;
    units::MevMass muonic_alpha;
    units::MevMass muonic_he3;
    //!@}

    //! Check whether all data are assigned
    CELER_FUNCTION explicit operator bool() const
    {
        return mu_minus > zero_quantity() && proton > zero_quantity()
               && triton > zero_quantity() && neutron > zero_quantity()
               && alpha > zero_quantity() && he3 > zero_quantity()
               && muonic_hydrogen > zero_quantity()
               && muonic_deuteron > zero_quantity()
               && muonic_triton > zero_quantity()
               && muonic_alpha > zero_quantity()
               && muonic_he3 > zero_quantity();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data for for the \c DTMixMucfModel .
 */
template<Ownership W, MemSpace M>
struct DTMixMucfData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using MaterialItems = Collection<T, W, M, MuCfMatId>;
    using GridRecord = NonuniformGridRecord;
    using CycleTimesArray = EnumArray<MucfMuonicMolecule, Array<real_type, 2>>;
    using MaterialFractionsArray = EnumArray<MucfIsotope, real_type>;

    //! Particles
    MucfParticleIds particle_ids;
    MucfParticleMasses particle_masses;

    //! Muon CDF energy grid for sampling outgoing muCF muons
    //! X-axis range is [0, 1) and y-axis is the outgoing muon energy in MeV
    GridRecord muon_energy_cdf;
    Items<real_type> reals;

    //!@{
    //! Material-dependent data calculated at model construction
    //! \c PhysMatId indexed by \c MuCfMatId
    MaterialItems<PhysMatId> mucfmatid_to_matid;
    //! Isotopic fractions per material: [mat_comp_id][isotope]
    MaterialItems<MaterialFractionsArray> isotopic_fractions;
    //! Cycle times per material: [mat_comp_id][muonic_molecule][spin_index]
    MaterialItems<CycleTimesArray> cycle_times;  //!< In [s]
    //! \todo Add mean atom spin flip times
    //! \todo Add mean atom transfer times
    //!@}

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return particle_ids && particle_masses && muon_energy_cdf
               && !reals.empty() && !mucfmatid_to_matid.empty()
               && !isotopic_fractions.empty() && !cycle_times.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    DTMixMucfData& operator=(DTMixMucfData<W2, M2> const& other)
    {
        CELER_EXPECT(other);

        //! \todo Finish implementation
        this->particle_ids = other.particle_ids;
        this->particle_masses = other.particle_masses;
        this->muon_energy_cdf = other.muon_energy_cdf;
        this->reals = other.reals;
        this->mucfmatid_to_matid = other.mucfmatid_to_matid;
        this->isotopic_fractions = other.isotopic_fractions;
        this->cycle_times = other.cycle_times;

        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
