//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/SimpleCmsTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "GeantTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness for "simple CMS".
 *
 * This geometry is a set of nested cylinders with length 1400 cm.
 *
 * | Radius [cm] | Material | Volume name |
 * | ----------: | -------- | ----------- |
 * |          0  |          |             |
 * |         30  | galactic | vacuum_tube |
 * |        125  | si       | si_tracker |
 * |        175  | pb       | em_calorimeter |
 * |        275  | c        | had_calorimeter |
 * |        375  | ti       | sc_solenoid |
 * |        700  | fe       | fe_muon_chambers |
 * |             | galactic | world |
 */
class SimpleCmsTestBase : virtual public GeantTestBase
{
  protected:
    std::string_view gdml_basename() const override { return "simple-cms"; }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
