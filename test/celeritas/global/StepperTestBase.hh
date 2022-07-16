//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/StepperTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <memory>
#include <vector>

#include "corecel/Types.hh"
#include "celeritas/GlobalTestBase.hh"

#include "DummyAction.hh"

namespace celeritas
{
struct StepperInput;
struct Primary;
class StepperInterface;
} // namespace celeritas

namespace celeritas_test
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
    class TestEm3Test : public celeritas_test::TestEm3Base,
                        public celeritas_test::StepperTestBase
    {
      public:
        //! Make 10GeV electrons along +x
        std::vector<Primary> make_primaries(size_type count) const override;
    };
 * \endcode
 */
class StepperTestBase : virtual public celeritas_test::GlobalTestBase
{
  public:
    //!@{
    //! \name Type aliases
    using size_type = celeritas::size_type;
    //!@}

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
    //!@{
    //! Whether the Geant4 configuration match a certain machine
    static bool is_ci_build();
    static bool is_wildstyle_build();
    static bool is_srj_build();
    //!@}

    // Add dummy action at construction
    StepperTestBase();

    // Construct the setup values for Stepper
    celeritas::StepperInput
    make_stepper_input(size_type tracks, size_type init_scaling);

    //! Create a vector of primaries inside the 'run' function
    virtual std::vector<celeritas::Primary>
    make_primaries(size_type count) const = 0;

    //! Maximum number of steps on average before aborting
    virtual size_type max_average_steps() const = 0;

    // Run a bunch of primaries to completion
    RunResult
    run(celeritas::StepperInterface& step, size_type num_primaries) const;

    //! Access the dummy action
    const DummyAction& dummy_action() const { return *dummy_action_; }

  protected:
    std::shared_ptr<DummyAction> dummy_action_;
};

//---------------------------------------------------------------------------//
//! Print the current configuration
struct PrintableBuildConf
{
};
std::ostream& operator<<(std::ostream& os, const PrintableBuildConf&);

//---------------------------------------------------------------------------//
} // namespace celeritas_test
