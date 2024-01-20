//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/StepperTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Types.hh"
#include "celeritas/GlobalTestBase.hh"

#include "DummyAction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct StepperInput;
struct Primary;
class StepperInterface;

namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Construct helper action and set up stepper/primary inputs.
 *
 * This class must be virtual so that it can be used as a mixin to other class
 * definitions.
 *
 * Example:
 * \code
    class TestEm3Test : public TestEm3Base,
                        public StepperTestBase
    {
      public:
        //! Make 10GeV electrons along +x
        std::vector<Primary> make_primaries(size_type count) const override;
    };
 * \endcode
 */
class StepperTestBase : virtual public GlobalTestBase
{
  public:
    struct SetupCheckResult
    {
        std::vector<std::string> processes;
        std::vector<std::string> actions;
        std::vector<std::string> actions_desc;

        // Print code for the expected attributes
        void print_expected() const;
    };

    // Whether the build uses the default real type and RNG
    static bool is_default_build();

    struct RunResult
    {
        using StepCount = std::pair<size_type, size_type>;

        std::vector<size_type> active;
        std::vector<size_type> queued;

        //! Total number of step iterations
        size_type num_step_iters() const { return active.size(); }
        // Cumulative number of steps over all tracks / number of primaries
        double calc_avg_steps_per_primary() const;
        // Index of first non-full step after capacity is reached
        size_type calc_emptying_step() const;
        // Step index and value of the high water mark of the initializer queue
        StepCount calc_queue_hwm() const;

        // Print code for the expected attributes
        void print_expected() const;

        //! True if run was performed and data is consistent
        explicit operator bool() const
        {
            return !active.empty() && queued.size() == active.size();
        }
    };

  public:
    // Add dummy action at construction
    StepperTestBase();

    // Construct the setup values for Stepper
    StepperInput make_stepper_input(size_type tracks);

    //! Create a vector of primaries inside the 'run' function
    virtual std::vector<Primary> make_primaries(size_type count) const = 0;

    //! Maximum number of steps on average before aborting
    virtual size_type max_average_steps() const = 0;

    // Get the physics output
    SetupCheckResult check_setup();

    // Run a bunch of primaries to completion
    RunResult run(StepperInterface& step, size_type num_primaries) const;

    //! Access the dummy action
    DummyAction const& dummy_action() const { return *dummy_action_; }

  protected:
    std::shared_ptr<DummyAction> dummy_action_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
