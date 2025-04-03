//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/AlongStepTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "corecel/Types.hh"
#include "celeritas/GlobalTestBase.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/phys/Interaction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<MemSpace>
class CoreState;
struct Primary;
class ExtendFromPrimariesAction;

namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Run one or more tracks with the same starting conditions for a single step.
 *
 * This high-level test *only* executes on the host so we can extract detailed
 * information from the states.
 */
class AlongStepTestBase : virtual public GlobalTestBase
{
  public:
    //!@{
    //! \name Type aliases
    using MevEnergy = units::MevEnergy;
    //!@}

    struct Input
    {
        ParticleId particle_id;
        MevEnergy energy{0};
        Real3 position{0, 0, 0};  // [cm]
        Real3 direction{0, 0, 1};
        real_type time{0};  // [s]
        real_type phys_mfp{1};  //!< Number of MFP to collision
        MscRange msc_range{0, 0, 0};

        explicit operator bool() const
        {
            return particle_id && energy >= zero_quantity() && time >= 0
                   && phys_mfp > 0;
        }
    };

    struct RunResult
    {
        real_type eloss{};  //!< Energy loss / MeV
        real_type displacement{};  //!< Distance from start to end points
        real_type angle{};  //!< Dot product of in/out direction
        real_type time{};  //!< Change in time
        real_type step{};  //!< Physical step length
        real_type mfp{};  //!< Number of MFP traveled over step
        real_type alive{};  //!< Fraction of tracks alive at end of step
        std::string action;  //!< Most likely action to take next

        void print_expected() const;
    };

    RunResult run(Input const&, size_type num_tracks = 1);

  private:
    void extend_from_primaries(Span<Primary const> primaries,
                               CoreState<MemSpace::host>* state);

    void
    execute_action(std::string const& label, CoreState<MemSpace::host>* state);

    void execute_action(CoreStepActionInterface const& action,
                        CoreState<MemSpace::host>* state) const;

    // Primary initialization
    std::shared_ptr<ExtendFromPrimariesAction const> primaries_action_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
