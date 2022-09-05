//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Main.hh
//---------------------------------------------------------------------------//
#pragma once

#include "testdetail/TestMain.hh"

//! Define main
int main(int argc, char** argv)
{
    return ::celeritas::testdetail::test_main(argc, argv);
}
