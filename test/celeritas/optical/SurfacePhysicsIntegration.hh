//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SurfacePhysicsIntegration.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Config.hh"

#include "corecel/cont/ArrayIO.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/sys/ActionGroups.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/KernelLauncher.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/GeantTestBase.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/TrackInitializer.hh"
#include "celeritas/optical/Transporter.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/gen/DirectGeneratorAction.hh"
#include "celeritas/optical/gen/OffloadData.hh"
#include "celeritas/optical/surface/SurfacePhysicsParams.hh"
#include "celeritas/phys/GeneratorRegistry.hh"
#include "celeritas/track/CoreStateCounters.hh"
#include "celeritas/track/TrackFunctors.hh"
#include "celeritas/track/Utils.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
/*!
 * Reference results:
 * - Double precision
 * - Orange geometry (requires valid surface normals and relocation on
 *   boundary)
 */
constexpr bool reference_configuration
    = ((CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
       && (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
       && (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW));

using namespace ::celeritas::test;
//---------------------------------------------------------------------------//
/*!
 */
template<class Collector>
class CollectResultsAction final : public OpticalStepActionInterface,
                                   public ConcreteAction
{
  public:
    explicit CollectResultsAction(ActionId aid, Collector& results)
        : ConcreteAction(aid, "collect-results", "collect test results")
        , results_(results)
    {
    }

    void step(CoreParams const& params, CoreStateHost& state) const final
    {
        for (auto tid : range(TrackSlotId{state.size()}))
        {
            CoreTrackView track(params.host_ref(), state.ref(), tid);
            auto sim = track.sim();
            if (this->is_post_boundary(track)
                || this->is_absorbed_on_boundary(track))
            {
                results_(track);
                sim.status(TrackStatus::killed);
            }
        }
    }

    void step(CoreParams const&, CoreStateDevice&) const final
    {
        CELER_NOT_IMPLEMENTED("not collecting on device");
    }

    StepActionOrder order() const final { return StepActionOrder::post; }

  private:
    //! Whether the track finished a boundary crossing
    inline bool is_post_boundary(CoreTrackView const& track) const
    {
        return AppliesValid{}(track)
               && track.sim().post_step_action()
                      == track.surface_physics().scalars().post_boundary_action;
    }

    //! Whether the track was absorbed during a boundary crossing
    inline bool is_absorbed_on_boundary(CoreTrackView const& track) const
    {
        return track.sim().status() == TrackStatus::killed
               && track.sim().post_step_action()
                      == track.surface_physics().scalars().surface_stepping_action;
    }

    Collector& results_;
};

//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    SurfacePhysicsIntegration ...;
   \endcode
 */
class SurfacePhysicsIntegrationTestBase : public GeantTestBase
{
  public:
    std::string_view gdml_basename() const override { return "optical-box"; }

    GeantPhysicsOptions build_geant_options() const override
    {
        auto result = GeantTestBase::build_geant_options();
        result.optical = {};
        CELER_ENSURE(result.optical);
        return result;
    }

    GeantImportDataSelection build_import_data_selection() const override
    {
        auto result = GeantTestBase::build_import_data_selection();
        result.processes |= GeantImportDataSelection::optical;
        return result;
    }

    std::vector<IMC> select_optical_models() const override
    {
        return {IMC::absorption};
    }

    void SetUp() override {}

    SPConstOpticalSurfacePhysics build_optical_surface_physics() override
    {
        inp::SurfacePhysics input;

        this->setup_surface_models(input);

        // Default surface

        PhysSurfaceId phys_surface = [&] {
            size_type num_surfaces = 0;
            for (auto const& mats : input.materials)
            {
                num_surfaces += mats.size() + 1;
            }
            return PhysSurfaceId(num_surfaces);
        }();

        input.materials.push_back({});
        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
        input.reflectivity.fresnel.emplace(phys_surface,
                                           inp::FresnelReflection{});
        input.interaction.trivial.emplace(phys_surface,
                                          TrivialInteractionMode::absorb);

        return std::make_shared<SurfacePhysicsParams>(
            this->optical_action_reg().get(), input);
    }

    void initialize_run()
    {
        auto generate = DirectGeneratorAction::make_and_insert(
            *this->core(), *this->optical_params());

        Transporter::Input inp;
        inp.params = this->optical_params();
        transport_ = std::make_shared<Transporter>(std::move(inp));

        size_type num_tracks = 128;
        auto state = std::make_shared<CoreState<MemSpace::host>>(
            *this->optical_params(), StreamId{0}, num_tracks);
        state->aux() = std::make_shared<AuxStateVec>(
            *this->core()->aux_reg(), MemSpace::host, StreamId{0}, num_tracks);
        state_ = state;
    }

    void run_step(real_type angle)
    {
        real_type sin_theta = std::sin(angle);
        real_type cos_theta = std::cos(angle);

        std::vector<TrackInitializer> inits(
            100,
            TrackInitializer{units::MevEnergy{3e-6},
                             from_cm(Real3{0, 49, 0}),
                             Real3{sin_theta, cos_theta, 0},
                             Real3{0, 0, 1},
                             0,
                             ImplVolumeId{0}});

        generate_->insert(*state_, make_span(inits));

        (*transport_)(*state_);
    }

  protected:
    std::shared_ptr<CoreState<MemSpace::host>> state_;
    std::shared_ptr<AuxStateVec> aux_;
    std::shared_ptr<Transporter> transport_;
    std::shared_ptr<DirectGeneratorAction> generate_;

    virtual void setup_surface_models(inp::SurfacePhysics&) const = 0;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
