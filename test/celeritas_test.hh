//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas_test.hh
//! Meta-include to facilitate testing.
//---------------------------------------------------------------------------//
#pragma once

#include <iostream>
#include <string_view>

#include "celeritas_config.h"

// IWYU pragma: begin_exports
#include "Test.hh"
#include "TestMacros.hh"
#include "TestMain.hh"
// IWYU pragma: end_exports

using std::cout;
using std::endl;
using namespace std::literals::string_view_literals;
