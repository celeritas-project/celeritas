#ifdef __ROOTCLING__
#pragma link C++ class G4VHit+;
#pragma link C++ class G4ThreeVector+;
#pragma link C++ class demo_geant::HitData+;
#pragma link C++ class demo_geant::HitRootEvent+;
#pragma link C++ class demo_geant::SensitiveHit+;
#pragma link C++ class std::vector<G4VHit*>+;
#pragma link C++ class std::vector<demo_geant::SensitiveHit*>+;
#pragma link C++ class std::map<std::string, std::vector<G4VHit*> >+;
#endif