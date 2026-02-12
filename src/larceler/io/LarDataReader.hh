//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/io/LarDataReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <lardataobj/Simulation/SimEnergyDeposit.h>

#include "celeritas/ext/RootFileManager.hh"
#include "celeritas/ext/RootUniquePtr.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Helper class to read ROOT files produced by the \c GeoAndSimDataExporter
 * module.
 */
class LarDataReader
{
  public:
    //!@{
    //! \name Type aliases
    using SPRootFileManager = std::shared_ptr<RootFileManager>;
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

  private:
    SPRootFileManager root_manager_;
    UPExtern<TTree> sim_tree_;

    // TTree branch references for sim::SimEnergyDeposit data
    struct SimEdepData
    {
        std::vector<int>* NumPhotons{nullptr};
        std::vector<int>* NumElectrons{nullptr};
        std::vector<double>* ScintYieldRatio{nullptr};
        std::vector<double>* Energy{nullptr};
        std::vector<double>* Time{nullptr};
        std::vector<double>* StartX{nullptr};
        std::vector<double>* StartY{nullptr};
        std::vector<double>* StartZ{nullptr};
        std::vector<double>* EndX{nullptr};
        std::vector<double>* EndY{nullptr};
        std::vector<double>* EndZ{nullptr};
        std::vector<double>* StartT{nullptr};
        std::vector<double>* EndT{nullptr};
        std::vector<int>* TrackID{nullptr};
        std::vector<int>* PdgCode{nullptr};
    } sim_edep_data_;

    //// HELPER FUNCTIONS ////

    // Hardcoded detector information tree name
    char const* detector_info_tree_name() const
    {
        return "data/detector_info";
    }

    // Hardcoded optical detector tree name
    char const* optical_detectors_tree_name() const
    {
        return "data/optical_detectors";
    }

    // Hardcoded sim::SimEnergyDeposit tree name
    char const* sim_data_tree_name() const
    {
        return "data/sim_energy_deposits";
    }
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
}  // namespace test
}  // namespace celeritas
