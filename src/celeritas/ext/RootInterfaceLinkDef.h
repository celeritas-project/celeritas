//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootInterfaceLinkDef.h
//! \brief Define the classes added to the ROOT dictionary
//---------------------------------------------------------------------------//
#ifdef __ROOTCLING__

// clang-format off
#pragma link C++ class celeritas::ImportParticle+;
#pragma link C++ class celeritas::ImportProcess+;
#pragma link C++ class celeritas::ImportModelMaterial+;
#pragma link C++ class celeritas::ImportModel+;
#pragma link C++ class celeritas::ImportMscModel+;
#pragma link C++ class celeritas::ImportPhysicsTable+;
#pragma link C++ class celeritas::ImportPhysicsVector+;
#pragma link C++ class celeritas::ImportMaterial+;
#pragma link C++ class celeritas::ImportProductionCut+;
#pragma link C++ class celeritas::ImportMatElemComponent+;
#pragma link C++ class celeritas::ImportElement+;
#pragma link C++ class celeritas::ImportIsotope+;
#pragma link C++ class celeritas::ImportVolume+;
#pragma link C++ class celeritas::ImportSBTable+;
#pragma link C++ class celeritas::ImportLivermoreSubshell+;
#pragma link C++ class celeritas::ImportLivermorePE+;
#pragma link C++ class celeritas::ImportAtomicTransition+;
#pragma link C++ class celeritas::ImportAtomicSubshell+;
#pragma link C++ class celeritas::ImportAtomicRelaxation+;
#pragma link C++ class celeritas::ImportEmParameters+;
#pragma link C++ class celeritas::ImportLoopingThreshold+;
#pragma link C++ class celeritas::ImportTransParameters+;
#pragma link C++ class celeritas::ImportData::ImportSBMap+;
#pragma link C++ class celeritas::ImportData::ImportLivermorePEMap+;
#pragma link C++ class celeritas::ImportData::ImportAtomicRelaxationMap+;
#pragma link C++ class celeritas::ImportData+;
// clang-format on

#endif
