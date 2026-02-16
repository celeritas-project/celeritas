//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarDataReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <lardataobj/Simulation/SimEnergyDeposit.h>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"  // IWYU pragma: keep

#include "SimEnergyDepositData.hh"

class TTree;
class TFile;
class TDirectory;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class to read ROOT files produced by the \c GeoSimExporterModule .
 *
 * \sa GeoSimExporterModule
 */
class LarDataReader
{
  public:
    //!@{
    //! \name Type aliases
    using VecSimEdep = std::vector<sim::SimEnergyDeposit>;
    using VecOpDetCenter = std::vector<Real3>;
    //!@}

  public:
    // Construct with ROOT file name
    explicit LarDataReader(std::string name);
    ~LarDataReader();
    CELER_DELETE_COPY_MOVE(LarDataReader);

    // Return number of events
    size_type num_events() const;

    // Return vector of SimEnergyDeposit objects for a given event ID
    VecSimEdep read_event(size_type event_id) const;

    // Return detector name
    std::string detector_name() const;

    // Return all optical detector centers, indexed by optical detector ID
    VecOpDetCenter optical_detector_centers() const;

    //!@{
    //! \name ROOT directory and tree name accessors
    // TDirectory name created by art; all TTrees are stored in this directory
    char const* data_dir_name() const { return "data"; }

    // Detector information tree name
    char const* detector_info_tree_name() const { return "detector_info"; }

    // Optical detector tree name
    char const* optical_detectors_tree_name() const
    {
        return "optical_detectors";
    }

    // SimEnergyDeposit data tree name
    char const* sim_data_tree_name() const { return "sim_energy_deposits"; }
    //!@}

  private:
    std::unique_ptr<TFile> root_file_;  //!< Input ROOT file
    std::unique_ptr<TDirectory> data_dir_;  //!< TDirectory with all TTrees
    std::unique_ptr<TTree> sim_tree_;  //!< SimEnergyDepositData TTree
    SimEnergyDepositData sim_edep_data_;  //!< SimEnergyDepositData refs
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
