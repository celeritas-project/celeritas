//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/CaloTestBase.cc
//---------------------------------------------------------------------------//
#include "CaloTestBase.hh"

#include <iostream>

#include "corecel/cont/Span.hh"
#include "corecel/io/OutputRegistry.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/user/SimpleCalo.hh"
#include "celeritas/user/StepCollector.hh"

using std::cout;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
CaloTestBase::~CaloTestBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct calorimeters and step collector at setup time.
 */
void CaloTestBase::SetUp()
{
    size_type num_streams = 1;

    std::vector<Label> labels;
    for (auto&& name : this->get_detector_names())
    {
        labels.push_back(name);
    }
    calo_ = std::make_shared<SimpleCalo>(
        std::move(labels), *this->geometry(), num_streams);

    StepCollector::VecInterface interfaces = {calo_};

    collector_ = std::make_shared<StepCollector>(std::move(interfaces),
                                                 this->geometry(),
                                                 num_streams,
                                                 this->action_reg().get());
}

//---------------------------------------------------------------------------//
//! Print the expected result
void CaloTestBase::RunResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static const double expected_edep[] = "
         << repr(this->edep)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_edep, result.edep);\n"
            "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
/*!
 * Run a number of tracks.
 */
auto CaloTestBase::run(size_type num_tracks, size_type num_steps) -> RunResult
{
    StepperInput step_inp;
    step_inp.params = this->core();
    step_inp.stream_id = StreamId{0};
    step_inp.num_track_slots = num_tracks;

    Stepper<MemSpace::host> step(step_inp);

    // Initial step
    auto primaries = this->make_primaries(num_tracks);
    auto count = step(make_span(primaries));

    while (count && --num_steps > 0)
    {
        count = step();
    }

    RunResult result;
    for (auto energy : calo_->calc_total_energy_deposition())
    {
        result.edep.push_back(energy.value());
    }
    calo_->clear();

    return result;
}
//---------------------------------------------------------------------------//
/*!
 * Get output from the example calorimeter.
 */
std::string CaloTestBase::output() const
{
    // See OutputInterface.hh
    return to_string(*calo_);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
