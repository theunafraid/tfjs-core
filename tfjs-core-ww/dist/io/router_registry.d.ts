import { IOHandler } from './types';
export declare type IORouter = (url: string | string[], onProgress?: Function) => IOHandler;
export declare class IORouterRegistry {
    private static instance;
    private saveRouters;
    private loadRouters;
    private constructor();
    private static getInstance;
    static registerSaveRouter(saveRouter: IORouter): void;
    static registerLoadRouter(loadRouter: IORouter): void;
    static getSaveHandlers(url: string | string[]): IOHandler[];
    static getLoadHandlers(url: string | string[], onProgress?: Function): IOHandler[];
    private static getHandlers;
}
export declare const registerSaveRouter: (loudRouter: IORouter) => void;
export declare const registerLoadRouter: (loudRouter: IORouter) => void;
export declare const getSaveHandlers: (url: string | string[]) => IOHandler[];
export declare const getLoadHandlers: (url: string | string[], onProgress?: Function) => IOHandler[];
