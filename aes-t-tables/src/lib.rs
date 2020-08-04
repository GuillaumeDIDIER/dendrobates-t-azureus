// Generic AES T-table attack flow

// Modularisation :
// The module handles loading, then passes the relevant target infos to a attack strategy object for calibration
// Then the module runs the attack, calling the attack strategy to make a measurement and return hit/miss

// interface for attack : run victim (eat a closure)
// interface for measure : give measurement target.

// Can attack strategies return err ?

// Load a vulnerable openssl - determine adresses af the T tables ?
// Run the calibrations
// Then start the attacks

// This is a serialized attack - either single threaded or synchronised

// parameters required

// an attacker measurement
// a calibration victim
