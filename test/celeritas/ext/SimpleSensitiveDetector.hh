//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/SimpleSensitiveDetector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>
#include <G4VSensitiveDetector.hh>

#include "celeritas/Types.hh"

#include "AssertionHelper.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct SimpleHitsResult
{
    static constexpr char class_str[] = "SimpleHitsResult";

    std::vector<double> energy_deposition;  // [MeV]
    std::vector<double> step_length;  // [cm]
    std::vector<std::string> particle;
    std::vector<double> weight;
    std::vector<int> track_id;
    std::vector<int> parent_id;
    std::vector<double> pre_energy;  // [MeV]
    std::vector<double> pre_time;  // [ns]
    std::vector<Real3> pre_pos;  // [cm]
    std::vector<Real3> pre_dir;
    std::vector<std::string> pre_physvol;
    std::vector<double> post_time;  // [ns]
    std::vector<double> post_energy;  // [MeV]
    std::vector<Real3> post_pos;  // [cm]
    std::vector<Real3> post_dir;
    std::vector<std::string> post_physvol;
    std::vector<std::string> post_status;

    friend std::ostream& operator<<(std::ostream&, SimpleHitsResult const&);
    friend void print_expected(SimpleHitsResult const&);
    friend void check_ref_eq(AssertionHelper&,
                             SimpleHitsResult const&,
                             SimpleHitsResult const&,
                             std::nullptr_t);
};

//---------------------------------------------------------------------------//
/*!
 * Store vectors of hit information.
 */
class SimpleSensitiveDetector final : public G4VSensitiveDetector
{
  public:
    // Construct with LV pointer for internal testing
    explicit SimpleSensitiveDetector(G4LogicalVolume const* lv);

    // Construct with string name when attaching to multiple LVs
    explicit SimpleSensitiveDetector(std::string const& name);

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
