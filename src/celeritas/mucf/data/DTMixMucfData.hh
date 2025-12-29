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
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * ParticleIds used by the \c DTMixMucfModel .
 */
struct MucfParticles
{
    //! Primary
    ParticleId mu_minus;

    //!@{
    //! Elementary particles and nuclei
    ParticleId neutron;
    ParticleId proton;
    ParticleId alpha;
    ParticleId helium3;
    //!@}

    //!@{
    //! Muonic atoms
    ParticleId muonic_hydrogen;
    ParticleId muonic_deuteron;
    ParticleId muonic_triton;
    ParticleId muonic_alpha;
    ParticleId muonic_helium3;
    //!@}

    //! Check whether all particles are assigned
    CELER_FUNCTION explicit operator bool() const
    {
        return mu_minus && neutron && proton && alpha && helium3
               && muonic_hydrogen && muonic_alpha && muonic_triton
               && muonic_helium3;
    }

    //! Assign from ParticleParams (anonymous free function in the model.cc?)
    static MucfParticles from_params(ParticleParams const& particles)
    {
#define MP_ADD(MEMBER)                               \
    pids.MEMBER = particles.find(pdg::MEMBER());     \
    if (!pids.MEMBER)                                \
    {                                                \
        missing.push_back({#MEMBER, pdg::MEMBER()}); \
    }

        using PairStrPdg = std::pair<std::string, PDGNumber>;
        std::vector<PairStrPdg> missing;
        MucfParticles pids;

        MP_ADD(mu_minus);
        MP_ADD(neutron);
        MP_ADD(proton);
        MP_ADD(alpha);
        //! \todo Decide whether to implement these PDGs in PDGNumber.hh
#if 0
        MP_ADD(helium_3);
        MP_ADD(muonic_hydrogen);
        MP_ADD(muonic_alpha);
        MP_ADD(muonic_deuteron);
        MP_ADD(muonic_triton);
        MP_ADD(muonic_helium3);
#endif
        CELER_VALIDATE(
            missing.empty(),
            << "missing particles required for muon-catalyzed fusion: "
            << join(missing.begin(), missing.end(), ", ", [](PairStrPdg const& p) {
                   return p.first + " (PDG "
                          + std::to_string(p.second.unchecked_get()) + ')';
               }));

        return pids;

#undef MP_ADD
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
    using MatCycleItems = Collection<T, W, M, MuCfMatCompId>;
    template<class T>
    using MaterialItems = Collection<T, W, M, MuCfMatCompId>;
    using Grid = NonuniformGridRecord;
    using CycleTimesArray = EnumArray<MucfMuonicMolecule, Array<real_type, 2>>;

    //! Particle IDs
    MucfParticles particles;

    //! Muon CDF energy grid for sampling outgoing muCF muons
    Grid muon_energy_cdf;  //! \todo Verify energy unit
    Items<real_type> reals;

    //!@{
    //! Material-dependent data calculated at model construction
    //! \c PhysMatId indexed by \c MuCfMatCompId
    MaterialItems<PhysMatId> matcompid_to_matid;
    //! Cycle times per material: [mat_comp_id][muonic_molecule][spin_index]
    MatCycleItems<CycleTimesArray> cycle_times;  //!< In [s]
    //! \todo Add mean atom spin flip times
    //! \todo Add mean atom transfer times
    //!@}

    //! \todo Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        //! \todo Finalize implementation
        return particles && muon_energy_cdf && !matcompid_to_matid.empty()
               && !cycle_times.empty()
               && (matcompid_to_matid.size() == cycle_times.size());
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
        this->matcompid_to_matid = other.matcompid_to_matid;
        this->cycle_times = other.cycle_times;

        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
