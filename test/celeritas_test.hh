//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas_test.hh
//! Meta-include to facilitate testing.
//---------------------------------------------------------------------------//
#pragma once

#include <iostream>
#include <string_view>

// IWYU pragma: begin_exports
#include "corecel/Config.hh"

#include "Test.hh"
#include "TestMacros.hh"
#include "TestMain.hh"
// IWYU pragma: end_exports

using std::cout;
using std::endl;
using namespace std::literals::string_view_literals;
