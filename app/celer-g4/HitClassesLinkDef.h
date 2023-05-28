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
#pragma link C++ class G4VHit+;
#pragma link C++ class G4ThreeVector+;
#pragma link C++ class celeritas::app::HitData+;
#pragma link C++ class celeritas::app::HitRootEvent+;
#pragma link C++ class celeritas::app::SensitiveHit+;
#pragma link C++ class std::vector<G4VHit*>+;
#pragma link C++ class std::vector<celeritas::app::SensitiveHit*>+;
#pragma link C++ class std::map<std::string, std::vector<G4VHit*> >+;
// clang-format on
#endif