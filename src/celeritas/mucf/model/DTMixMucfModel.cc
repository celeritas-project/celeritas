//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/model/DTMixMucfModel.cc
//---------------------------------------------------------------------------//
#include "DTMixMucfModel.hh"

#include <algorithm>

#include "corecel/OpaqueIdIO.hh"
#include "corecel/inp/Grid.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/grid/NonuniformGridBuilder.hh"
#include "celeritas/inp/MucfPhysics.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/mucf/executor/DTMixMucfExecutor.hh"  // IWYU pragma: associated
#include "celeritas/phys/InteractionApplier.hh"  // IWYU pragma: associated
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "detail/MucfMaterialInserter.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Assign particle IDs from \c ParticleParams .
 */
static std::pair<MucfParticleIds, MucfParticleMasses>
from_params(ParticleParams const& particles)
{
    using PairStrPdg = std::pair<std::string, PDGNumber>;
    std::vector<PairStrPdg> missing;
    MucfParticleIds ids;
    MucfParticleMasses masses;
#define MP_ADD(MEMBER)                               \
    ids.MEMBER = particles.find(pdg::MEMBER());      \
    if (!ids.MEMBER)                                 \
    {                                                \
        missing.push_back({#MEMBER, pdg::MEMBER()}); \
    }                                                \
    else                                             \
    {                                                \
        auto p_view = particles.get(ids.MEMBER);     \
        masses.MEMBER = p_view.mass();               \
    }

    MP_ADD(mu_minus);
    MP_ADD(neutron);
    MP_ADD(proton);
    MP_ADD(alpha);
    MP_ADD(he3);
    MP_ADD(muonic_deuteron);
    MP_ADD(muonic_triton);
    MP_ADD(muonic_alpha);

    //! \todo Decide whether to implement these PDGs in PDGNumber.hh
#if 0
    MP_ADD(muonic_hydrogen);
    MP_ADD(muonic_he3);
#endif

    CELER_VALIDATE(missing.empty(),
                   << "missing particles required for muon-catalyzed fusion: "
                   << join_stream(missing.begin(),
                                  missing.end(),
                                  ", ",
                                  [](std::ostream& os, PairStrPdg const& p) {
                                      os << p.first << " (PDG "
                                         << p.second.unchecked_get() << ')';
                                  }));
    return {ids, masses};

#undef MP_ADD
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 *
 * Most of the muon-catalyzed fusion data is static throughout the simulation,
 * as it is only material-dependent (DT mixture and temperature). Therefore,
 * most grids can be host-only and used to calculate final values, which are
 * then cached and copied to device. The exception to this is the muon energy
 * CDF grid, needed to sample the final state of the outgoing muon after a muCF
 * interaction.
 *
 * \todo Correctly update \c ImportProcessClass and \c ImportModelClass . These
 * operate under the assumption that there is a one-to-one equivalente between
 * Geant4 and Celeritas. But for muCF, everything is done via one
 * process/model/executor in Celeritas, whereas in Geant4 atom formation, spin
 * flip, atom transfer, etc., are are all separate processes.
 */
DTMixMucfModel::DTMixMucfModel(ActionId id,
                               ParticleParams const& particles,
                               MaterialParams const& materials)
    : StaticConcreteAction(
          id,
          "dt-mucf",
          R"(interact by muon forming and fusing a dd, dt, or tt muonic molecule)")
{
    CELER_EXPECT(id);

    // Initialize muCF physics input data
    inp::MucfPhysics inp_data = inp::MucfPhysics::from_default();
    CELER_EXPECT(inp_data);

    HostVal<DTMixMucfData> host_data;
    auto [ids, masses] = from_params(particles);
    host_data.particle_ids = ids;
    host_data.particle_masses = masses;

    // Copy muon energy CDF data using NonuniformGridBuilder
    NonuniformGridBuilder build_grid_record{&host_data.reals};
    host_data.muon_energy_cdf = build_grid_record(inp_data.muon_energy_cdf);

    // Calculate and cache quantities for all materials with dt mixtures
    detail::MucfMaterialInserter insert(&host_data);
    for (auto const& matid : range(materials.num_materials()))
    {
        auto const& mat_view = materials.get(PhysMatId{matid});
        if (insert(mat_view))
        {
            CELER_LOG(debug) << "Added material ID " << mat_view.material_id()
                             << " as a muCF d-t mixture";
        }
    }

    // Copy to device
    data_ = ParamsDataStore<DTMixMucfData>{std::move(host_data)};
    CELER_ENSURE(this->data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto DTMixMucfModel::applicability() const -> SetApplicability
{
    Applicability applic;
    applic.particle = this->host_ref().particle_ids.mu_minus;
    // At-rest model
    applic.lower = zero_quantity();
    applic.upper = zero_quantity();

    return {applic};
}

//---------------------------------------------------------------------------//
/*!
 * At-rest model does not require microscopic cross sections.
 */
auto DTMixMucfModel::micro_xs(Applicability) const -> XsTable
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Interact with host data.
 */
void DTMixMucfModel::step(CoreParams const& params, CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{DTMixMucfExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void DTMixMucfModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
