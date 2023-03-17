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
template<MemSpace M>
auto CaloTestBase::run(size_type num_tracks,
                       size_type num_steps,
                       size_type num_batches) -> RunResult
{
    // Can't have fewer than 1 track per batch
    CELER_ASSERT(num_batches<=num_tracks);

    // Don't want to deal with remainders
    CELER_ASSERT(num_tracks%num_batches==0);

    // Compute tracks per batch
    size_type num_tracks_per_batch=num_tracks/num_batches;

    // Initialize RunResult
    RunResult result;
    size_t num_detectors = this->calo_->num_detectors();
    result.edep=std::vector<double>(num_detectors,0.);
    result.edep_err=std::vector<double>(num_detectors,0.);

    // Loop over batches
    for(size_type i_batch=0; i_batch<num_batches; ++i_batch){

      this->run_impl<M>(num_tracks_per_batch, num_steps);

      // Retrieve energies deposited this batch for each detector
      auto edep = calo_->calc_total_energy_deposition();

      // Update results for each detector
      for(size_t i_det=0; i_det<num_detectors; ++i_det){
        auto edep_det=edep[i_det];
        result.edep.at(i_det)+=edep_det;
        result.edep_err.at(i_det)+=(edep_det*edep_det);
      }

      calo_->clear();
    }

    // Finally, compute the mean and relative_err
    double norm=num_batches > 1 ?  1.0/double(num_batches) : 1.0;
    for(size_t i_det=0; i_det<num_detectors; ++i_det){
      auto mu=result.edep.at(i_det)*norm;
      auto var=result.edep_err.at(i_det)*norm - mu*mu;
      auto err=sqrt(var) / mu;
      result.edep.at(i_det) = mu;
      result.edep_err.at(i_det) = err;
    }

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
    CaloTestBase::run<MemSpace::device>(size_type, size_type, size_type);
template CaloTestBase::RunResult
    CaloTestBase::run<MemSpace::host>(size_type, size_type, size_type);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
