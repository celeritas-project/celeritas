//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/SimpleSensitiveDetector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>
#include <G4VSensitiveDetector.hh>

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct SimpleHitsResult
{
    std::vector<double> energy_deposition;  // [MeV]
    std::vector<double> step_length;  // [cm]
    std::vector<std::string> particle;
    std::vector<double> pre_energy;  // [MeV]
    std::vector<double> pre_pos;  // [cm]
    std::vector<std::string> pre_physvol;
    std::vector<double> post_time;  // [ns]

    void print_expected() const;
};

//---------------------------------------------------------------------------//
/*!
 * Store vectors of hit information.
 */
class SimpleSensitiveDetector final : public G4VSensitiveDetector
{
  public:
    explicit SimpleSensitiveDetector(G4LogicalVolume const* lv);

    //! Access hit data
    SimpleHitsResult const& hits() const { return hits_; }

    //! Reset hits between tests
    void clear() { hits_ = {}; }

    //! Get the logical volume this SD is attached to
    G4LogicalVolume const* lv() const { return lv_; }

  protected:
    void Initialize(G4HCofThisEvent*) final { this->clear(); }
    bool ProcessHits(G4Step*, G4TouchableHistory*) final;

  private:
    SimpleHitsResult hits_;
    G4LogicalVolume const* lv_{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
