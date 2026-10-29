//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventIOTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <functional>
#include <iosfwd>

#include "corecel/io/Repr.hh"
#include "celeritas/io/EventIOInterface.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "AssertionHelper.hh"
#include "Test.hh"
#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Helpers for testing reading+writing of event files.
 *
 * This defines several particles: proton, d_quark, anti_u_quark, w_minus,
 * gamma; and provides utilities for reading all primaries from a file.
 */
class EventIOTestBase : public Test
{
  public:
    //!@{
    //! \name Type aliases
    using VecPrimary = std::vector<Primary>;
    using Reader = EventReaderInterface;
    using Writer = EventWriterInterface;
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    //!@}

  protected:
    struct ReadAllResult
    {
        static constexpr char class_str[] = "ReadAllResult";

        struct Tolerance
        {
            double energy{Test::fine_eps};
            double pos{Test::fine_eps};
            double dir{Test::fine_eps};
            double time{Test::fine_eps};
        };

        std::vector<int> pdg;
        std::vector<double> energy;
        std::vector<Real3> pos;
        std::vector<Real3> dir;
        std::vector<double> time;
        std::vector<int> event;

        friend std::ostream& operator<<(std::ostream& os,
                                        ReadAllResult const& ref)
        {
            return os << '{' << CELER_ATTR_REPR(ref, pdg) << ", "
                      << CELER_ATTR_REPR(ref, energy) << ", "
                      << CELER_ATTR_REPR(ref, pos) << ", "
                      << CELER_ATTR_REPR(ref, dir) << ", "
                      << CELER_ATTR_REPR(ref, time) << ", "
                      << CELER_ATTR_REPR(ref, event) << '}';
        }

        friend void check_ref_eq(AssertionHelper& helper,
                                 ReadAllResult const& expected,
                                 ReadAllResult const& actual,
                                 std::nullptr_t)
        {
            check_ref_eq(helper, expected, actual, Tolerance{});
        }

        friend void check_ref_eq(AssertionHelper& helper,
                                 ReadAllResult const& expected,
                                 ReadAllResult const& actual,
                                 Tolerance const& tol)
        {
            using namespace ::celeritas::test;
            is_vec_eq(helper, "pdg", expected.pdg, actual.pdg);
            is_vec_soft_eq(
                helper, "energy", expected.energy, actual.energy, tol.energy);
            is_vec_soft_eq(helper, "pos", expected.pos, actual.pos, tol.pos);
            is_vec_soft_eq(helper, "dir", expected.dir, actual.dir, tol.dir);
            is_vec_soft_eq(
                helper, "time", expected.time, actual.time, tol.time);
            is_vec_eq(helper, "event", expected.event, actual.event);
        }
    };

    // Create particles
    void SetUp() override;

    // Read all primaries from a file
    ReadAllResult read_all(Reader& read_event) const;

    // Write test primaries to a file
    void write_test_event(Writer& write_event) const;
    // Read and check the test primaries to a file
    void read_check_test_event(Reader& read_event) const;

    // Access particles
    SPConstParticles const& particles() const { return particles_; }

  private:
    SPConstParticles particles_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
