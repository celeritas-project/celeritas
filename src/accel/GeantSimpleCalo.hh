//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/GeantSimpleCalo.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "corecel/io/OutputInterface.hh"
#include "celeritas/Quantities.hh"

class G4LogicalVolume;
class G4VSensitiveDetector;

namespace celeritas
{
//---------------------------------------------------------------------------//
class SharedParams;
namespace detail
{
struct GeantSimpleCaloStorage;
}

//---------------------------------------------------------------------------//
/*!
 * Manage a simple calorimeter sensitive detector across threads.
 *
 * The factory should be created in DetectorConstruction or
 * DetectorConstruction::Construct and added to the output parameters. Calling
 * \c MakeSensitiveDetector will emit a sensitive detector for the local thread
 * *and attach it* to the logical volumes on the local thread.
 */
class GeantSimpleCalo final : public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParams = std::shared_ptr<SharedParams const>;
    using UPSensitiveDetector = std::unique_ptr<G4VSensitiveDetector>;
    using VecLV = std::vector<G4LogicalVolume*>;
    using VecReal = std::vector<double>;
    using EnergyUnits = units::Mev;
    //!@}

  public:
    // Construct with SD name and the volumes to attach the SD to
    GeantSimpleCalo(std::string name, SPConstParams params, VecLV volumes);

    // Emit a new detector for the local thread and attach to the stored LVs
    UPSensitiveDetector MakeSensitiveDetector();

    //! Get the list of volumes with this SD attached
    VecLV const& Volumes() const { return volumes_; }

    // Get accumulated energy deposition over all threads
    VecReal CalcTotalEnergyDeposition() const;

    //!@{
    //! \name Output interface

    //! Category of data to write
    Category category() const final { return Category::result; }
    // Key for the entry inside the category
    std::string_view label() const final;
    // Write output to the given JSON object
    void output(JsonPimpl*) const final;
    //!@}

  private:
    using SPStorage = std::shared_ptr<detail::GeantSimpleCaloStorage>;
    SPConstParams params_;
    VecLV volumes_;
    SPStorage storage_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
