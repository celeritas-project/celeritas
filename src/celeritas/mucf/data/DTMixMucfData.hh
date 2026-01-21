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
    ParticleId neutron;
    ParticleId proton;
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
        return mu_minus && neutron && proton && alpha && he3 && muonic_hydrogen
               && muonic_alpha && muonic_triton && muonic_he3;
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

    //! Particle IDs
    MucfParticleIds particles;

    //! Muon CDF energy grid for sampling outgoing muCF muons
    GridRecord muon_energy_cdf;  //! \todo Verify energy unit
    Items<real_type> reals;

    //!@{
    //! Material-dependent data calculated at model construction
    //! \c PhysMatId indexed by \c MuCfMatId
    MaterialItems<PhysMatId> mucfmatid_to_matid;
    //! Cycle times per material: [mat_comp_id][muonic_molecule][spin_index]
    MaterialItems<CycleTimesArray> cycle_times;  //!< In [s]
    //! \todo Add mean atom spin flip times
    //! \todo Add mean atom transfer times
    //!@}

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return particles && muon_energy_cdf && !mucfmatid_to_matid.empty()
               && !cycle_times.empty()
               && (mucfmatid_to_matid.size() == cycle_times.size());
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    DTMixMucfData& operator=(DTMixMucfData<W2, M2> const& other)
    {
        CELER_EXPECT(other);

        //! \todo Finish implementation
        this->particles = other.particles;
        this->reals = other.reals;
        this->muon_energy_cdf = other.muon_energy_cdf;
        this->mucfmatid_to_matid = other.mucfmatid_to_matid;
        this->cycle_times = other.cycle_times;

        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
