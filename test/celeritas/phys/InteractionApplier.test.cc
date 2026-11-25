//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/InteractionApplier.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/phys/InteractionApplier.hh"

#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/SimpleTestBase.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Secondary.hh"
#include "celeritas/track/TrackInitData.hh"

#include "celeritas_test.hh"

using celeritas::units::MevEnergy;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Return a predetermined interaction
struct MockInteract
{
    Interaction result;

    Interaction operator()(CoreTrackView const&) const { return result; }
};

//---------------------------------------------------------------------------//

class InteractionApplierTest : public SimpleTestBase
{
  protected:
    struct Result
    {
        real_type energy_deposition{};
        std::vector<bool> survived;
        size_type num_cut{};
    };

    SPConstParticle build_particle() override
    {
        using namespace units;

        ParticleParams::Input inp;
        inp.push_back({"gamma",
                       pdg::gamma(),
                       zero_quantity(),
                       zero_quantity(),
                       constants::stable_decay_constant});
        inp.push_back({"electron",
                       pdg::electron(),
                       MevMass{0.5},
                       ElementaryCharge{-1},
                       constants::stable_decay_constant});
        inp.push_back({"positron",
                       pdg::positron(),
                       MevMass{0.5},
                       ElementaryCharge{1},
                       constants::stable_decay_constant});
        return std::make_shared<ParticleParams>(std::move(inp));
    }

    SPConstCutoff build_cutoff() override
    {
        CutoffParams::Input inp;
        inp.materials = this->material();
        inp.particles = this->particle();
        inp.cutoffs = {
            {pdg::gamma(), {{MevEnergy{0.1}, 0.1}, {}}},
            {pdg::electron(), {{MevEnergy{1}, 0.1}, {}}},
            {pdg::positron(), {{MevEnergy{1}, 0.1}, {}}},
        };
        inp.apply_post_interaction = apply_cuts_;
        return std::make_shared<CutoffParams>(std::move(inp));
    }

    Secondary make_secondary(PDGNumber pdg, real_type energy)
    {
        Secondary result;
        result.particle_id = this->particle()->find(pdg);
        result.energy = MevEnergy{energy};
        result.direction = {0, 0, 1};
        return result;
    }

    //! Apply a 10 MeV photon absorption with the given weight and secondaries
    Result run(real_type weight,
               real_type local_deposition,
               std::vector<Secondary> secondaries)
    {
        CoreState<MemSpace::host> state{*this->core(), StreamId{0}, 1};
        CoreTrackView track{
            this->core()->host_ref(), state.ref(), TrackSlotId{0}};

        // Initialize a photon inside the aluminum box
        TrackInitializer init;
        init.sim.track_id = TrackId{0};
        init.sim.primary_id = PrimaryId{0};
        init.sim.event_id = EventId{0};
        init.sim.weight = weight;
        init.geo = {from_cm(Real3{0, 0, 0}), Real3{0, 0, 1}};
        init.particle.particle_id = this->particle()->find(pdg::gamma());
        init.particle.energy = MevEnergy{10};
        track = init;
        track.sim().status(TrackStatus::alive);
        track.physics_step().reset_energy_deposition();
        track.physics_step().secondaries({});

        // Apply interaction
        Interaction interaction = Interaction::from_absorption();
        interaction.energy_deposition = MevEnergy{local_deposition};
        interaction.secondaries = make_span(secondaries);
        InteractionApplier apply{MockInteract{interaction}};
        apply(track);

        EXPECT_EQ(TrackStatus::killed, track.sim().status());
        EXPECT_REAL_EQ(weight, track.sim().weight());

        Result result;
        result.energy_deposition
            = value_as<MevEnergy>(track.physics_step().energy_deposition());
        for (Secondary const& s : track.physics_step().secondaries())
        {
            result.survived.push_back(static_cast<bool>(s));
        }
        result.num_cut = track.counters().num_cut;
        return result;
    }

    bool apply_cuts_{true};
};

//---------------------------------------------------------------------------//
// Energy deposition is weighted by scoring, so it cannot depend on the weight
TEST_F(InteractionApplierTest, cut_secondaries)
{
    for (real_type weight : {0.5, 1.0, 10.0})
    {
        SCOPED_TRACE(weight);

        // No secondaries are below the cutoffs
        auto result = this->run(weight,
                                0.25,
                                {this->make_secondary(pdg::electron(), 2),
                                 this->make_secondary(pdg::gamma(), 0.2)});
        EXPECT_SOFT_EQ(0.25, result.energy_deposition);
        EXPECT_EQ((std::vector<bool>{true, true}), result.survived);
        EXPECT_EQ(0, result.num_cut);

        // Only the below-cutoff electron is deposited
        result = this->run(weight,
                           0.25,
                           {this->make_secondary(pdg::electron(), 0.5),
                            this->make_secondary(pdg::gamma(), 0.2)});
        EXPECT_SOFT_EQ(0.75, result.energy_deposition);
        EXPECT_EQ((std::vector<bool>{false, true}), result.survived);
        EXPECT_EQ(1, result.num_cut);

        // Below-cutoff positron deposits its rest mass and its annihilation
        // partner's
        result = this->run(weight,
                           0.25,
                           {this->make_secondary(pdg::positron(), 0.5),
                            this->make_secondary(pdg::gamma(), 0.05)});
        EXPECT_SOFT_EQ(0.25 + (0.5 + 2 * 0.5) + 0.05, result.energy_deposition);
        EXPECT_EQ((std::vector<bool>{false, false}), result.survived);
        EXPECT_EQ(2, result.num_cut);
    }
}

TEST_F(InteractionApplierTest, disabled_cuts)
{
    apply_cuts_ = false;
    auto result
        = this->run(10, 0.25, {this->make_secondary(pdg::electron(), 0.5)});
    EXPECT_SOFT_EQ(0.25, result.energy_deposition);
    EXPECT_EQ((std::vector<bool>{true}), result.survived);
    EXPECT_EQ(0, result.num_cut);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
