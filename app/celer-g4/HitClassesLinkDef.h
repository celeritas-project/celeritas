//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file app/celer-g4/HitClassesLinkDef.h
//! \brief Request ROOT for classes used in app/celer-g4
//---------------------------------------------------------------------------//
#ifdef __ROOTCLING__
// clang-format off
#pragma link C++ class celeritas::app::HitData+;
#pragma link C++ class celeritas::app::HitEventData+;
#pragma link C++ class std::vector<celeritas::app::HitData>+;
// clang-format on
#endif