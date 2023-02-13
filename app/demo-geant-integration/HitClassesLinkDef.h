//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file app/demo-geant-integration/HitClassesLinkDef.h
//! \brief Request ROOT for classes used in app/demo-geant-integration
//---------------------------------------------------------------------------//
#ifdef __ROOTCLING__
// clang-format off
#pragma link C++ class G4VHit+;
#pragma link C++ class G4ThreeVector+;
#pragma link C++ class demo_geant::HitData+;
#pragma link C++ class demo_geant::HitRootEvent+;
#pragma link C++ class demo_geant::SensitiveHit+;
#pragma link C++ class std::vector<G4VHit*>+;
#pragma link C++ class std::vector<demo_geant::SensitiveHit*>+;
#pragma link C++ class std::map<std::string, std::vector<G4VHit*> >+;
// clang-format on
#endif