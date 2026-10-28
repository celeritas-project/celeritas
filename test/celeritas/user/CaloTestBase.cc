//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/CaloTestBase.cc
//---------------------------------------------------------------------------//
#include "CaloTestBase.hh"

#include <iostream>

#include "corecel/cont/Span.hh"
#include "celeritas/geo/CoreGeoParams.hh"
#include "celeritas/user/SimpleCalo.hh"
#include "celeritas/user/StepCollector.hh"

#include "AssertionHelper.hh"
#include "testdetail/TestMacrosImpl.hh"

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
    std::vector<Label> labels;
    for (auto&& name : this->get_detector_names())
    {
        labels.push_back(name);
    }

    // Make sure geometry is constructed
    this->geometry();

    size_type const num_streams = 1;
    calo_ = std::make_shared<SimpleCalo>(
        std::move(labels), num_streams, *this->volume());

    StepCollector::VecInterface interfaces = {calo_};

    collector_
        = StepCollector::make_and_insert(*this->core(), std::move(interfaces));
}

//---------------------------------------------------------------------------//
//! Print the expected result
void CaloTestBase::RunResult::print_expected() const
{
    ::celeritas::testdetail::print_expected(*this);
}

std::ostream& operator<<(std::ostream& os, CaloTestBase::RunResult const& ref)
{
    return os << '{' << repr(ref.edep) << '}';
}

void check_ref_eq(AssertionHelper& helper,
                  CaloTestBase::RunResult const& expected,
                  CaloTestBase::RunResult const& actual,
                  std::nullptr_t)
{
    check_ref_eq(
        helper, expected, actual, CaloTestBase::RunResult::Tolerance{});
}

void check_ref_eq(AssertionHelper& helper,
                  CaloTestBase::RunResult const& expected,
                  CaloTestBase::RunResult const& actual,
                  CaloTestBase::RunResult::Tolerance const& tol)
{
    is_vec_soft_eq(helper, "edep", expected.edep, actual.edep, tol.edep);
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
template CaloTestBase::RunResult CaloTestBase::run<MemSpace::device>(
    size_type, size_type);
template CaloTestBase::RunResult CaloTestBase::run<MemSpace::host>(size_type,
                                                                   size_type);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
