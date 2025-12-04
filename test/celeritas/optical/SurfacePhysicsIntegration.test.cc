//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SurfacePhysicsIntegration.test.cc
//---------------------------------------------------------------------------//
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

#include "celeritas_test.hh"

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
 * Counters for photon status after a run at a single angle.
 */
struct CollectResults
{
    size_type num_absorbed{0};
    size_type num_failed{0};
    size_type num_reflected{0};
    size_type num_refracted{0};

    //! Clear counters
    void reset()
    {
        num_absorbed = 0;
        num_failed = 0;
        num_reflected = 0;
        num_refracted = 0;
    }

    //! Score track
    void operator()(CoreTrackView const& track)
    {
        if (track.sim().status() == TrackStatus::alive)
        {
            auto vol = track.geometry().volume_instance_id();
            if (vol == VolumeInstanceId{1})
            {
                num_reflected++;
                return;
            }
            else if (vol == VolumeInstanceId{2})
            {
                num_refracted++;
                return;
            }
        }
        else if (track.sim().status() == TrackStatus::killed)
        {
            num_absorbed++;
            return;
        }

        num_failed++;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Hook into the stepping loop to score and kill optical tracks that finish
 * doing a surface crossing.
 */
class CollectResultsAction final : public OpticalStepActionInterface,
                                   public ConcreteAction
{
  public:
    explicit CollectResultsAction(ActionId aid, CollectResults& results)
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

    CollectResults& results_;
};

//---------------------------------------------------------------------------//
/*!
 * Counter results for a series of runs at different angles.
 */
struct SurfaceTestResults
{
    std::vector<size_type> num_absorbed;
    std::vector<size_type> num_reflected;
    std::vector<size_type> num_refracted;
};

//---------------------------------------------------------------------------//
// TEST CHASSIS
//---------------------------------------------------------------------------//

class SurfacePhysicsIntegrationTest : public GeantTestBase
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

    virtual void setup_surface_models(inp::SurfacePhysics&) const = 0;

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

    void build_state(size_type num_tracks)
    {
        auto state = std::make_shared<CoreState<MemSpace::host>>(
            *this->optical_params(), StreamId{0}, num_tracks);
        state->aux() = std::make_shared<AuxStateVec>(
            *this->core()->aux_reg(), MemSpace::host, StreamId{0}, num_tracks);
        state_ = state;
    }

    void make_collector()
    {
        auto& reg = *this->optical_params()->action_reg();
        auto collector
            = std::make_shared<CollectResultsAction>(reg.next_id(), collect_);
        reg.insert(collector);
    }

    void build_transporter()
    {
        Transporter::Input inp;
        inp.params = this->optical_params();
        transport_ = std::make_shared<Transporter>(std::move(inp));
    }

    //! Run over a set of angles and collect the results
    SurfaceTestResults run(std::vector<real_type> const& angles)
    {
        auto generate = DirectGeneratorAction::make_and_insert(
            *this->core(), *this->optical_params());

        this->make_collector();
        this->build_transporter();
        this->build_state(128);

        SurfaceTestResults results;
        for (auto deg_angle : angles)
        {
            collect_.reset();

            auto angle = deg_angle * constants::pi / 180;
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

            generate->insert(*state_, make_span(inits));

            (*transport_)(*state_);

            EXPECT_EQ(0, collect_.num_failed);
            results.num_absorbed.push_back(collect_.num_absorbed);
            results.num_reflected.push_back(collect_.num_reflected);
            results.num_refracted.push_back(collect_.num_refracted);
        }

        return results;
    }

  protected:
    std::shared_ptr<CoreState<MemSpace::host>> state_;
    std::shared_ptr<AuxStateVec> aux_;
    std::shared_ptr<Transporter> transport_;
    CollectResults collect_;
};

//---------------------------------------------------------------------------//
class SurfacePhysicsIntegrationBackscatterTest
    : public SurfacePhysicsIntegrationTest
{
  public:
    void setup_surface_models(inp::SurfacePhysics& input) const final
    {
        PhysSurfaceId phys_surface{0};

        // center-top surface

        input.materials.push_back({});
        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
        input.reflectivity.fresnel.emplace(phys_surface,
                                           inp::FresnelReflection{});

        // Only back-scattering

        input.interaction.trivial.emplace(phys_surface,
                                          TrivialInteractionMode::backscatter);
    }
};

//---------------------------------------------------------------------------//
class SurfacePhysicsIntegrationAbsorbTest : public SurfacePhysicsIntegrationTest
{
  public:
    void setup_surface_models(inp::SurfacePhysics& input) const final
    {
        PhysSurfaceId phys_surface{0};

        // center-top surface

        input.materials.push_back({});
        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
        input.reflectivity.fresnel.emplace(phys_surface,
                                           inp::FresnelReflection{});

        // Only absorption

        input.interaction.trivial.emplace(phys_surface,
                                          TrivialInteractionMode::absorb);
    }
};

//---------------------------------------------------------------------------//
class SurfacePhysicsIntegrationTransmitTest
    : public SurfacePhysicsIntegrationTest
{
  public:
    void setup_surface_models(inp::SurfacePhysics& input) const final
    {
        PhysSurfaceId phys_surface{0};

        // center-top surface

        input.materials.push_back({});
        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
        input.reflectivity.fresnel.emplace(phys_surface,
                                           inp::FresnelReflection{});

        // Only transmission

        input.interaction.trivial.emplace(phys_surface,
                                          TrivialInteractionMode::transmit);
    }
};

//---------------------------------------------------------------------------//
class SurfacePhysicsIntegrationFresnelTest
    : public SurfacePhysicsIntegrationTest
{
  public:
    void setup_surface_models(inp::SurfacePhysics& input) const final
    {
        PhysSurfaceId phys_surface{0};

        // center-top surface

        input.materials.push_back({});
        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
        input.reflectivity.fresnel.emplace(phys_surface,
                                           inp::FresnelReflection{});

        // Fresnel refraction / reflection

        input.interaction.dielectric.emplace(
            phys_surface,
            inp::DielectricInteraction::from_dielectric(
                inp::ReflectionForm::from_spike()));
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Only back-scattering
TEST_F(SurfacePhysicsIntegrationBackscatterTest, backscatter)
{
    if (reference_configuration)
    {
        std::vector<real_type> angles{0, 30, 60};
        auto result = this->run(angles);

        SurfaceTestResults expected;
        expected.num_reflected = {100, 100, 100};
        expected.num_refracted = {0, 0, 0};
        expected.num_absorbed = {0, 0, 0};

        EXPECT_EQ(expected.num_reflected, result.num_reflected);
        EXPECT_EQ(expected.num_refracted, result.num_refracted);
        EXPECT_EQ(expected.num_absorbed, result.num_absorbed);
    }
}

//---------------------------------------------------------------------------//
// Only absorption
TEST_F(SurfacePhysicsIntegrationAbsorbTest, absorb)
{
    if (reference_configuration)
    {
        std::vector<real_type> angles{0, 30, 60};
        auto result = this->run(angles);

        SurfaceTestResults expected;
        expected.num_refracted = {0, 0, 0};
        expected.num_reflected = {0, 0, 0};
        expected.num_absorbed = {100, 100, 100};

        EXPECT_EQ(expected.num_reflected, result.num_reflected);
        EXPECT_EQ(expected.num_refracted, result.num_refracted);
        EXPECT_EQ(expected.num_absorbed, result.num_absorbed);
    }
}

//---------------------------------------------------------------------------//
// Only transmission
TEST_F(SurfacePhysicsIntegrationTransmitTest, transmit)
{
    if (reference_configuration)
    {
        std::vector<real_type> angles{0, 30, 60};
        auto result = this->run(angles);

        SurfaceTestResults expected;
        expected.num_refracted = {100, 100, 100};
        expected.num_reflected = {0, 0, 0};
        expected.num_absorbed = {0, 0, 0};

        EXPECT_EQ(expected.num_reflected, result.num_reflected);
        EXPECT_EQ(expected.num_refracted, result.num_refracted);
        EXPECT_EQ(expected.num_absorbed, result.num_absorbed);
    }
}

//---------------------------------------------------------------------------//
// Fresnel reflection / refraction
TEST_F(SurfacePhysicsIntegrationFresnelTest, fresnel)
{
    if (reference_configuration)
    {
        std::vector<real_type> angles{
            0,
            10,
            20,
            30,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            60,
            70,
            80,
        };

        auto result = this->run(angles);

        static unsigned int const expected_num_absorbed[] = {
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
            0u,
        };
        static unsigned int const expected_num_reflected[] = {
            2u,
            0u,
            3u,
            4u,
            15u,
            11u,
            9u,
            17u,
            18u,
            34u,
            27u,
            42u,
            60u,
            100u,
            100u,
            100u,
            100u,
            100u,
        };
        static unsigned int const expected_num_refracted[] = {
            98u,
            100u,
            97u,
            96u,
            85u,
            89u,
            91u,
            83u,
            82u,
            66u,
            73u,
            58u,
            40u,
            0u,
            0u,
            0u,
            0u,
            0u,
        };

        EXPECT_VEC_EQ(expected_num_reflected, result.num_reflected);
        EXPECT_VEC_EQ(expected_num_refracted, result.num_refracted);
        EXPECT_VEC_EQ(expected_num_absorbed, result.num_absorbed);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
