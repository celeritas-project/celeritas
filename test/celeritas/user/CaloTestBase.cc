//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/CaloTestBase.cc
//---------------------------------------------------------------------------//
#include "CaloTestBase.hh"

#include <iostream>

#include "corecel/cont/Span.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/user/SimpleCalo.hh"
#include "celeritas/user/StepCollector.hh"

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
    using std::cout;

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
template<MemSpace M>
auto CaloTestBase::run(size_type num_tracks, size_type num_steps) -> RunResult
{
    this->run_impl<M>(num_tracks, num_steps);

    auto edep = calo_->calc_total_energy_deposition();
    RunResult result;
    result.edep.assign(edep.begin(), edep.end());
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
template CaloTestBase::RunResult
    CaloTestBase::run<MemSpace::device>(size_type, size_type);
template CaloTestBase::RunResult
    CaloTestBase::run<MemSpace::host>(size_type, size_type);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
