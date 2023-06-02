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
//! Initalize results
void CaloTestBase::initalize()
{
    // Initialize RunResult vectors
    size_t num_detectors = this->calo_->num_detectors();
    result.edep=std::vector<double>(num_detectors,0.);
    result.edep_err=std::vector<double>(num_detectors,0.);
}

//---------------------------------------------------------------------------//
//! Gather results during a batch
void CaloTestBase::gather_batch_results()
{
  // Retrieve energies deposited this batch for each detector
  auto edep = calo_->calc_total_energy_deposition();

  // Update results for each detector
  size_t num_detectors = this->calo_->num_detectors();
  for(size_t i_det=0; i_det<num_detectors; ++i_det){
    auto edep_det=edep[i_det];
    result.edep.at(i_det)+=edep_det;
    result.edep_err.at(i_det)+=(edep_det*edep_det);
  }
  calo_->clear();
}

//---------------------------------------------------------------------------//
//! Finalize results
void CaloTestBase::finalize()
{
  if ( num_batches_ <= 1 ) return;

  // Compute the mean and relative_err over batches for each detector
  double norm= 1.0/double(num_batches_);
  size_t num_detectors = this->calo_->num_detectors();
  for(size_t i_det=0; i_det<num_detectors; ++i_det){
    auto mu=result.edep.at(i_det)*norm;
    auto var=result.edep_err.at(i_det)*norm- mu*mu;
      CELER_ASSERT(var>0);
      auto err=sqrt(var) / mu;
      result.edep.at(i_det) = mu;
      result.edep_err.at(i_det) = err;
  }
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
   this->run_impl<M>(num_tracks, num_steps, num_batches);
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
