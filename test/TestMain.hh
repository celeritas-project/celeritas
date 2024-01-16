//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TestMain.hh
//---------------------------------------------------------------------------//
#pragma once

#include "testdetail/TestMainImpl.hh"

//! Define main
int main(int argc, char** argv)
{
    return ::celeritas::testdetail::test_main(argc, argv);
}
