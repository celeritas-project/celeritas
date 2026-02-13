//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarDataReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/ext/RootUniquePtr.hh"

#include "SimEnergyDepositData.hh"

namespace sim
{
class SimEnergyDeposit;
}  // namespace sim

namespace celeritas
{
// Forward declare ROOT classes
class TFile;
class TDirectory;
class TTree;

//---------------------------------------------------------------------------//
/*!
 * Helper class to read ROOT files produced by the \c GeoAndSimDataExporter
 * module.
 *
 * \sa GeoAndSimDataExporter
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
    UPExtern<TFile> root_file_;
    UPExtern<TDirectory> data_dir_;  //!< TDirectory with all TTrees
    UPExtern<TTree> sim_tree_;  //!< TTree with SimEnergyDeposit input data
    SimEnergyDepositData sim_edep_data_;  //!<  TBranch data references
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline LarDataReader::LarDataReader(std::string name)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline size_type LarDataReader::num_events() const
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline LarDataReader::VecSimEdep
LarDataReader::read_event(size_type event_id) const
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline std::string LarDataReader::detector_name() const
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline LarDataReader::VecOpDetCenter
LarDataReader::optical_detector_centers() const
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
